/*
 * holomoe.c - HoloMoE: run a Qwen3.5/3.6 hybrid mixture-of-experts model from
 * a HoloMoE plate (built by shardpress), streaming experts from the SSD.
 *
 *   holomoe [PLATE_DIR]                 chat (default ~/holomoe/plate)
 *   holomoe PLATE -p "question"         one answer, then exit
 *
 * Architecture (from the official reference code):
 *   - 3 of every 4 layers: Gated DeltaNet linear attention. A fixed 128x128
 *     memory per head that decays and updates every word: constant memory.
 *   - every 4th layer: gated full attention with partial rotary positions.
 *   - every layer: 256 routed experts (top 8) + one always-on shared expert.
 * Experts are ternary bitplanes in the plate (popcount math), memory-mapped
 * zero-copy and paged in from disk on demand; the OS keeps hot ones in RAM.
 */
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

typedef uint16_t bf16;
static void die(const char *m) { fprintf(stderr, "holomoe: %s\n", m); exit(1); }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s); if (!p) die("out of memory"); return p; }
static inline float bf2f(bf16 h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, u;
    if (!e) { if (!m) u = s; else { e = 113; while (!(m & 1024)) { m <<= 1; e--; } u = s | (e << 23) | ((m & 1023) << 13); } }
    else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static double now(void) { struct timeval t; gettimeofday(&t, NULL); return t.tv_sec + t.tv_usec * 1e-6; }

/* ------------------------------------------------------------------ tokenizer
 * Byte-level BPE (Qwen2 / tiktoken style): split text into chunks with the
 * Qwen pre-tokenizer rules, then repeatedly merge the adjacent pair whose
 * joined bytes form the token with the lowest merge rank. */

typedef struct { const uint8_t *s; uint32_t len; int32_t rank; } Tok;
static Tok *TK; static int NTOK; static int *SORTED, NSORTED; static int BYTE_ID[256];

static int tokcmp(const uint8_t *a, uint32_t al, const uint8_t *b, uint32_t bl) {
    int c = memcmp(a, b, al < bl ? al : bl);
    return c ? c : (int)al - (int)bl;
}
static int sort_cb(const void *x, const void *y) {
    const Tok *a = &TK[*(const int *)x], *b = &TK[*(const int *)y];
    return tokcmp(a->s, a->len, b->s, b->len);
}
/* walk the flattened 256-adic tree: binary search over prefix order */
static int tok_lookup(const uint8_t *s, uint32_t len) {
    int lo = 0, hi = NSORTED - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, id = SORTED[mid];
        int c = tokcmp(TK[id].s, TK[id].len, s, len);
        if (!c) return id;
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static void vocab_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "holomoe: missing %s (run holomoe_meta.py)\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = xcalloc(1, (size_t)n); if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("vocab read");
    fclose(f);
    if (memcmp(buf, "HMVB", 4)) die("bad vocab.bin");
    uint32_t nt; memcpy(&nt, buf + 4, 4);
    NTOK = (int)nt; TK = xcalloc(nt, sizeof(Tok));
    size_t off = 8;
    for (uint32_t i = 0; i < nt; i++) {
        uint32_t len; int32_t r;
        memcpy(&len, buf + off, 4); memcpy(&r, buf + off + 4, 4); off += 8;
        TK[i].s = buf + off; TK[i].len = len; TK[i].rank = r; off += len;
    }
    SORTED = xcalloc(nt, sizeof(int));
    for (int i = 0; i < NTOK; i++) if (TK[i].rank != -2 && TK[i].len) SORTED[NSORTED++] = i;
    qsort(SORTED, NSORTED, sizeof(int), sort_cb);
    for (int b = 0; b < 256; b++) { uint8_t c = (uint8_t)b; BYTE_ID[b] = tok_lookup(&c, 1); }
}

static int is_nl(uint8_t c) { return c == '\n' || c == '\r'; }
static int is_sp(uint8_t c) { return c == ' ' || c == '\t' || c == '\v' || c == '\f' || is_nl(c); }
static int is_dg(uint8_t c) { return c >= '0' && c <= '9'; }
static int is_lt(uint8_t c) { return isalpha(c) || c >= 0x80; }   /* non-ASCII counted as letters */

/* length of the next pre-token starting at s[i] (Qwen2 split regex) */
static size_t pretok(const uint8_t *s, size_t n, size_t i) {
    uint8_t c = s[i];
    if (c == '\'' && i + 1 < n) {                                  /* 's 't 're 've 'm 'll 'd */
        uint8_t a = (uint8_t)tolower(s[i + 1]), b = i + 2 < n ? (uint8_t)tolower(s[i + 2]) : 0;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }
    size_t j = i;                                                  /* [^\r\n L N]? L+ */
    if (!is_lt(c) && !is_dg(c) && !is_nl(c) && i + 1 < n && is_lt(s[i + 1])) j++;
    if (j < n && is_lt(s[j])) { while (j < n && is_lt(s[j])) j++; return j - i; }
    if (is_dg(c)) return 1;                                        /* \p{N} */
    j = i + (c == ' ' ? 1 : 0);                                    /*  ?[^\s L N]+[\r\n]* */
    if (j < n && !is_sp(s[j]) && !is_lt(s[j]) && !is_dg(s[j])) {
        while (j < n && !is_sp(s[j]) && !is_lt(s[j]) && !is_dg(s[j])) j++;
        while (j < n && is_nl(s[j])) j++;
        return j - i;
    }
    j = i; while (j < n && is_sp(s[j])) j++;                       /* whitespace runs */
    size_t last_nl = 0; int has_nl = 0;
    for (size_t k = i; k < j; k++) if (is_nl(s[k])) { last_nl = k; has_nl = 1; }
    if (has_nl) return last_nl + 1 - i;                            /* \s*[\r\n]+ */
    if (j < n && j - i > 1) return j - i - 1;                      /* \s+(?!\S) */
    return j - i;                                                  /* \s+ */
}

static int encode(const char *text, int *out, int max) {
    const uint8_t *s = (const uint8_t *)text; size_t n = strlen(text), i = 0; int cnt = 0;
    uint32_t *st = xcalloc(n + 1, 4), *ln = xcalloc(n + 1, 4); int *id = xcalloc(n + 1, sizeof(int));
    while (i < n) {
        size_t k = pretok(s, n, i); if (!k) k = 1;
        int m = 0;
        for (size_t b = i; b < i + k; b++) { st[m] = (uint32_t)b; ln[m] = 1; id[m] = BYTE_ID[s[b]]; m++; }
        for (;;) {                                                 /* merge lowest rank pair */
            int best = -1, bid = -1; int32_t br = INT32_MAX;
            for (int p = 0; p + 1 < m; p++) {
                int t = tok_lookup(s + st[p], ln[p] + ln[p + 1]);
                if (t >= 0 && TK[t].rank >= 0 && TK[t].rank < br) { br = TK[t].rank; best = p; bid = t; }
            }
            if (best < 0) break;
            ln[best] += ln[best + 1]; id[best] = bid;
            memmove(&st[best + 1], &st[best + 2], (size_t)(m - best - 2) * 4);
            memmove(&ln[best + 1], &ln[best + 2], (size_t)(m - best - 2) * 4);
            memmove(&id[best + 1], &id[best + 2], (size_t)(m - best - 2) * sizeof(int));
            m--;
        }
        for (int p = 0; p < m && cnt < max; p++) out[cnt++] = id[p];
        i += k;
    }
    free(st); free(ln); free(id);
    return cnt;
}

static void put_tok(int id, int escape) {
    if (id < 0 || id >= NTOK) return;
    for (uint32_t i = 0; i < TK[id].len; i++) {
        uint8_t c = TK[id].s[i];
        if (escape && c == '\n') fputs("\\n", stdout); else putchar(c);
    }
}

/* ------------------------------------------------------------------ config */

static int DIM, NL, NH, NKV, HD, ROT, NE, TOPK, MOEI, SHI, VOCAB, LKH, LVH, LKD, LVD, CONVK;
static float THETA, EPS;
static char LTYPES[256];
static int NORM_OFF = 1, ATTN_GATE = 1, DENSE_I, NORM_TOPK = 1;   /* family adapters: Qwen3.6 / Qwen3.5 / Qwen3 */
static char ARCH[64] = "qwen3_5_moe", PFX[64];
static int IM_START = -1, IM_END = -1, EOT = -1, THINK = -1, THINK_END = -1;

static void cfg_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "holomoe: missing %s (run holomoe_meta.py)\n", path); exit(1); }
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '='); if (!eq) continue;
        *eq = 0; char *k = line, *v = eq + 1; v[strcspn(v, "\r\n")] = 0;
#define I(name, var) if (!strcmp(k, name)) var = atoi(v);
        I("dim", DIM) I("layers", NL) I("n_heads", NH) I("n_kv", NKV) I("head_dim", HD) I("rot_dim", ROT)
        I("n_exp", NE) I("top_k", TOPK) I("moe_i", MOEI) I("shared_i", SHI) I("vocab", VOCAB)
        I("lin_kh", LKH) I("lin_vh", LVH) I("lin_kd", LKD) I("lin_vd", LVD) I("conv_k", CONVK)
        I("im_start", IM_START) I("im_end", IM_END) I("eot", EOT) I("think", THINK) I("think_end", THINK_END)
        I("norm_off", NORM_OFF) I("attn_gate", ATTN_GATE) I("dense_i", DENSE_I) I("norm_topk", NORM_TOPK)
#undef I
        if (!strcmp(k, "theta")) THETA = (float)atof(v);
        if (!strcmp(k, "eps")) EPS = (float)atof(v);
        if (!strcmp(k, "arch")) snprintf(ARCH, sizeof ARCH, "%s", v);
        if (!strcmp(k, "layer_types")) snprintf(LTYPES, sizeof LTYPES, "%s", v);
    }
    fclose(f);
    if (!DIM || !NL || (int)strlen(LTYPES) < NL) die("model.cfg incomplete");
    if (!NE && !DENSE_I) die("model.cfg has neither experts nor a dense MLP size");
}

/* ------------------------------------------------------------------ plate pieces (zero-copy) */

typedef struct { const void *data; int dtype; uint64_t shape[8]; int ndim; uint64_t nbytes; const float *rs; } Dense;   /* dtype 0 f32, 1 bf16, 2 f16, 3 int8 rows, 4 int4 rows */
static char PLATE[4096];

static const void *map_file(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY); if (fd < 0) return NULL;
    struct stat sb; fstat(fd, &sb);
    void *m = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return NULL;
    if (len) *len = (size_t)sb.st_size;
    return m;
}

static double DENSE_PARAMS;                 /* every non-expert weight, counted from the plate */
static Dense dense(const char *name, int required) {
    Dense d; memset(&d, 0, sizeof d);
    char path[8192]; snprintf(path, sizeof path, "%s/dense/%s.t", PLATE, name);
    size_t len; const uint8_t *m = map_file(path, &len);
    if (!m) { if (required) { fprintf(stderr, "holomoe: missing tensor %s\n", name); exit(1); } return d; }
    if (memcmp(m, "HMD1", 4) || len < 104) die("bad dense piece");
    uint32_t h[4]; memcpy(h, m, 16); d.dtype = (int)h[1]; d.ndim = (int)h[2];
    memcpy(d.shape, m + 16, 64); memcpy(&d.nbytes, m + 80, 8);
    d.data = m + 88;
    { double n = 1; for (int i = 0; i < d.ndim; i++) n *= (double)d.shape[i]; DENSE_PARAMS += n; }
    return d;
}

static float *to_f32(Dense d, size_t n) {
    float *o = xcalloc(n, 4);
    for (size_t i = 0; i < n; i++)
        o[i] = d.dtype == 0 ? ((const float *)d.data)[i] : d.dtype == 2 ? h2f(((const uint16_t *)d.data)[i]) : bf2f(((const bf16 *)d.data)[i]);
    return o;
}


/* ------------------------------------------------------------------ math */

#define NCH 64
typedef void (*ChunkFn)(void *, size_t);
static void parfor(void *ctx, ChunkFn fn) {
#if defined(__APPLE__)
    dispatch_apply_f(NCH, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ctx, fn);
#else
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NCH; i++) fn(ctx, (size_t)i);
#endif
}

static double T_DENSE, T_EXP, T_ATT, T_HEAD;   /* where the time goes (/stats) */

/* int8 copy of a dense matrix, one scale per 32 weights (Q8_0 style) */
static int Q8 = 1; static double Q8_BYTES;
typedef struct { Dense *d; int8_t *q; float *rs; size_t rows, cols; } QJ;
static void q8_chunk(void *c, size_t i) {
    QJ *j = c; size_t per = (j->rows + NCH - 1) / NCH, a = i * per, b = a + per; if (b > j->rows) b = j->rows;
    for (size_t r = a; r < b; r++)
        for (size_t g = 0; g < j->cols; g += 32) {
            float tmp[32], mx = 0;
            for (size_t k = 0; k < 32; k++) {
                size_t ix = r * j->cols + g + k;
                tmp[k] = j->d->dtype == 0 ? ((const float *)j->d->data)[ix] : bf2f(((const bf16 *)j->d->data)[ix]);
                if (fabsf(tmp[k]) > mx) mx = fabsf(tmp[k]);
            }
            float s = mx > 0 ? mx / 127.0f : 1.0f; j->rs[(r * j->cols + g) / 32] = s;
            for (size_t k = 0; k < 32; k++) j->q[r * j->cols + g + k] = (int8_t)lrintf(tmp[k] / s);
        }
}
static volatile uint8_t SINK;
static void touch(const uint8_t *p, size_t n) {
    long pg = sysconf(_SC_PAGESIZE); uint8_t s = 0;
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(pg - 1);
    madvise((void *)a, n + ((uintptr_t)p - a), MADV_WILLNEED);
    for (size_t i = 0; i < n; i += (size_t)pg) s ^= p[i];
    s ^= p[n - 1]; SINK ^= s;
}
static double Q8_BUILT;                 /* bytes built this boot (first boot only) */
static int q8_dir_ok = -1;
static void q8ify(Dense *d, const char *name) {
    if (!Q8 || !d->data || d->ndim != 2 || (d->dtype != 0 && d->dtype != 1) || d->shape[1] % 32) return;
    size_t rows = d->shape[0], cols = d->shape[1], nq = rows * cols, ns = nq / 32, want = 64 + nq + ns * 4;
    char path[8192], tmp[8300];
    snprintf(path, sizeof path, "%s/q8/%s.q8", PLATE, name); snprintf(tmp, sizeof tmp, "%s.part", path);
    size_t len = 0; const uint8_t *m = map_file(path, &len);           /* saved copy from an earlier boot: map it, done */
    if (m && len == want && !memcmp(m, "HMQ8", 4) && ((const uint64_t *)m)[1] == rows && ((const uint64_t *)m)[2] == cols) {
        d->data = m + 64; d->rs = (const float *)(m + 64 + nq); d->dtype = 3; d->nbytes = nq; Q8_BYTES += (double)want;
        touch(m, len);                                                   /* pull it into RAM now, not mid-sentence */
        return;
    }
    if (m) munmap((void *)m, len);
    QJ j = {d, xcalloc(nq, 1), xcalloc(ns, 4), rows, cols};             /* first boot: build it ... */
    parfor(&j, q8_chunk);
    long pg = sysconf(_SC_PAGESIZE); uintptr_t a = (uintptr_t)d->data & ~(uintptr_t)(pg - 1);
    madvise((void *)a, (size_t)d->nbytes + ((uintptr_t)d->data - a), MADV_DONTNEED);
    d->data = j.q; d->rs = j.rs; d->dtype = 3; d->nbytes = nq; Q8_BYTES += (double)want; Q8_BUILT += (double)want;
    if (q8_dir_ok < 0) { char dir[8192]; snprintf(dir, sizeof dir, "%s/q8", PLATE); q8_dir_ok = mkdir(dir, 0755) == 0 || errno == EEXIST; }
    if (!q8_dir_ok) return;
    FILE *f = fopen(tmp, "wb"); if (!f) return;                          /* ... and save it: .part, flush, rename */
    uint64_t h[8] = {0}; memcpy(h, "HMQ8", 4); ((uint32_t *)h)[1] = 1; h[1] = rows; h[2] = cols;
    int ok = fwrite(h, 8, 8, f) == 8 && fwrite(j.q, 1, nq, f) == nq && fwrite(j.rs, 4, ns, f) == ns;
    ok = fflush(f) == 0 && fsync(fileno(f)) == 0 && ok; ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp, path) != 0) { unlink(tmp); if (q8_dir_ok == 1) { fprintf(stderr, "\n[could not save int8 copy (disk full?) - it will be rebuilt next boot]\n"); q8_dir_ok = 0; } }
}

/* 4-bit copy (lean mode): 32 weights per group, scale picked to minimise error, cached in plate/q4/ */
static int LEAN = 0; static int q4_dir_ok = -1;
typedef struct { Dense *d; uint8_t *q; float *rs; size_t rows, cols; } Q4J;
static void q4_chunk(void *c, size_t i) {
    Q4J *j = c; size_t per = (j->rows + NCH - 1) / NCH, a = i * per, b = a + per; if (b > j->rows) b = j->rows;
    static const float f[] = {1.0f, 0.95f, 0.9f, 0.85f, 0.8f, 0.75f};
    for (size_t r = a; r < b; r++)
        for (size_t g = 0; g < j->cols; g += 32) {
            float w[32], mx = 0;
            for (int k = 0; k < 32; k++) {
                size_t ix = r * j->cols + g + k;
                w[k] = j->d->dtype == 0 ? ((const float *)j->d->data)[ix] : bf2f(((const bf16 *)j->d->data)[ix]);
                if (fabsf(w[k]) > mx) mx = fabsf(w[k]);
            }
            float best_s = mx > 0 ? mx / 7.0f : 1.0f, best_e = 1e30f;
            for (int c = 0; c < 6 && mx > 0; c++) {               /* a smaller scale often fits the bulk of weights better */
                float sc = mx / 7.0f * f[c], e = 0;
                for (int k = 0; k < 32; k++) { int q = (int)lrintf(w[k] / sc); if (q > 7) q = 7; if (q < -8) q = -8; float d2 = w[k] - q * sc; e += d2 * d2; }
                if (e < best_e) { best_e = e; best_s = sc; }
            }
            j->rs[(r * j->cols + g) / 32] = best_s;
            uint8_t *o = j->q + (r * j->cols + g) / 2;
            for (int k = 0; k < 16; k++) {
                int lo = (int)lrintf(w[k] / best_s), hi = (int)lrintf(w[k + 16] / best_s);
                lo = lo > 7 ? 7 : lo < -8 ? -8 : lo; hi = hi > 7 ? 7 : hi < -8 ? -8 : hi;
                o[k] = (uint8_t)((lo + 8) | ((hi + 8) << 4));
            }
        }
}
static void q4ify(Dense *d, const char *name) {
    if (!d->data || d->ndim != 2 || (d->dtype != 0 && d->dtype != 1) || d->shape[1] % 32) return;
    size_t rows = d->shape[0], cols = d->shape[1], nq = rows * cols / 2, ns = rows * cols / 32, want = 64 + nq + ns * 4;
    char path[8192], tmp[8300];
    snprintf(path, sizeof path, "%s/q4/%s.q4", PLATE, name); snprintf(tmp, sizeof tmp, "%s.part", path);
    size_t len = 0; const uint8_t *m = map_file(path, &len);
    if (m && len == want && !memcmp(m, "HMQ4", 4) && ((const uint64_t *)m)[1] == rows && ((const uint64_t *)m)[2] == cols) {
        d->data = m + 64; d->rs = (const float *)(m + 64 + nq); d->dtype = 4; d->nbytes = nq; Q8_BYTES += (double)want;
        touch(m, len); return;
    }
    if (m) munmap((void *)m, len);
    Q4J j = {d, xcalloc(nq, 1), xcalloc(ns, 4), rows, cols};
    parfor(&j, q4_chunk);
    d->data = j.q; d->rs = j.rs; d->dtype = 4; d->nbytes = nq; Q8_BYTES += (double)want; Q8_BUILT += (double)want;
    if (q4_dir_ok < 0) { char dir[8192]; snprintf(dir, sizeof dir, "%s/q4", PLATE); q4_dir_ok = mkdir(dir, 0755) == 0 || errno == EEXIST; }
    if (!q4_dir_ok) return;
    FILE *f = fopen(tmp, "wb"); if (!f) return;
    uint64_t h[8] = {0}; memcpy(h, "HMQ4", 4); h[1] = rows; h[2] = cols;
    int ok = fwrite(h, 8, 8, f) == 8 && fwrite(j.q, 1, nq, f) == nq && fwrite(j.rs, 4, ns, f) == ns;
    ok = fflush(f) == 0 && fsync(fileno(f)) == 0 && ok; ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp, path) != 0) unlink(tmp);
}

/* activations quantized the same way, so the inner loop is int8 x int8 (vector units do 16-32 at once) */
static int8_t *AQ; static float *AS; static size_t AQN;
static void quant_act(const float *x, int B, int d) {
    size_t need = (size_t)B * d;
    if (need > AQN) { free(AQ); free(AS); AQ = xcalloc(need, 1); AS = xcalloc(need / 32 + 1, 4); AQN = need; }
    for (size_t g = 0; g < need; g += 32) {
        float mx = 0; for (int k = 0; k < 32; k++) if (fabsf(x[g + k]) > mx) mx = fabsf(x[g + k]);
        float s = mx > 0 ? mx / 127.0f : 1.0f, inv = 1.0f / s; AS[g / 32] = s;
        for (int k = 0; k < 32; k++) AQ[g + k] = (int8_t)lrintf(x[g + k] * inv);
    }
}
#if defined(__AVX2__) && defined(__FMA__)
static int SIMD = 1;
static inline float dot_q8q8_avx2(const int8_t *w, const float *ws, const int8_t *x, const float *xs, int d) {
    __m256 acc = _mm256_setzero_ps(); const __m256i one = _mm256_set1_epi16(1);
    for (int g = 0; g < d; g += 32) {                       /* 32 int8 products per instruction group */
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + g)), xv = _mm256_loadu_si256((const __m256i *)(x + g));
        __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(xv, xv), _mm256_sign_epi8(wv, xv));   /* |x| * (w * sign x) */
        __m256 f = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, one));
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(ws[g / 32] * xs[g / 32]), acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
#else
static int SIMD = 0;
#endif
/* 4-bit: 32 weights per group = 16 bytes (low nibbles = first 16, high = next 16), one float scale */
#if defined(__AVX2__) && defined(__FMA__)
static inline float dot_q4q8_avx2(const uint8_t *w, const float *ws, const int8_t *x, const float *xs, int d) {
    __m256 acc = _mm256_setzero_ps(); const __m256i one = _mm256_set1_epi16(1);
    const __m128i m4 = _mm_set1_epi8(15); const __m256i e8 = _mm256_set1_epi8(8);
    for (int g = 0; g < d; g += 32) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + g / 2));
        __m256i wv = _mm256_sub_epi8(_mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(b, 4), m4), _mm_and_si128(b, m4)), e8);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + g));
        __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(xv, xv), _mm256_sign_epi8(wv, xv));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p16, one)), _mm256_set1_ps(ws[g / 32] * xs[g / 32]), acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
#endif
static int SIMD_Q4 = 1;
static inline float dot_q4q8(const uint8_t *w, const float *ws, const int8_t *x, const float *xs, int d) {
#if defined(__AVX2__) && defined(__FMA__)
    if (SIMD_Q4) return dot_q4q8_avx2(w, ws, x, xs, d);
#endif
    float s = 0;
    for (int g = 0; g < d; g += 32) {
        const uint8_t *b = w + g / 2; int acc = 0;
        for (int k = 0; k < 16; k++) acc += ((b[k] & 15) - 8) * x[g + k] + ((b[k] >> 4) - 8) * x[g + 16 + k];
        s += (float)acc * ws[g / 32] * xs[g / 32];
    }
    return s;
}
static inline float dot_q8q8(const int8_t *w, const float *ws, const int8_t *x, const float *xs, int d) {
#if defined(__AVX2__) && defined(__FMA__)
    if (SIMD) return dot_q8q8_avx2(w, ws, x, xs, d);
#endif
    float s = 0;
    for (int g = 0; g < d; g += 32) {
        int acc = 0;
        for (int k = 0; k < 32; k++) acc += (int)w[g + k] * (int)x[g + k];
        s += (float)acc * ws[g / 32] * xs[g / 32];
    }
    return s;
}
static inline float dot_bf(const bf16 *w, const float *x, int d) {
    float acc[8] = {0}; int j = 0;
    for (; j + 8 <= d; j += 8) for (int k = 0; k < 8; k++) acc[k] += bf2f(w[j + k]) * x[j + k];
    float s = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    for (; j < d; j++) s += bf2f(w[j]) * x[j];
    return s;
}

/* o[B][n] = W[n][d] * x[B][d]: every weight row is read once for the whole batch */
typedef struct { float *o; Dense W; const float *x; int B, n, d; } MM;
static void mm_chunk(void *c, size_t i) {
    MM *m = c; int per = (m->n + NCH - 1) / NCH, a = (int)i * per, b = a + per; if (b > m->n) b = m->n;
    for (int r = a; r < b; r++) {
        if (m->W.dtype == 4) {
            const uint8_t *w = (const uint8_t *)m->W.data + (size_t)r * (m->d / 2); const float *ws = m->W.rs + (size_t)r * (m->d / 32);
            for (int t = 0; t < m->B; t++) m->o[(size_t)t * m->n + r] = dot_q4q8(w, ws, AQ + (size_t)t * m->d, AS + (size_t)t * (m->d / 32), m->d);
        } else if (m->W.dtype == 3) {
            const int8_t *w = (const int8_t *)m->W.data + (size_t)r * m->d; const float *ws = m->W.rs + (size_t)r * (m->d / 32);
            for (int t = 0; t < m->B; t++) m->o[(size_t)t * m->n + r] = dot_q8q8(w, ws, AQ + (size_t)t * m->d, AS + (size_t)t * (m->d / 32), m->d);
        } else if (m->W.dtype == 0) {
            const float *w = (const float *)m->W.data + (size_t)r * m->d;
            for (int t = 0; t < m->B; t++) { const float *x = m->x + (size_t)t * m->d; float s = 0; for (int j = 0; j < m->d; j++) s += w[j] * x[j]; m->o[(size_t)t * m->n + r] = s; }
        } else {
            const bf16 *w = (const bf16 *)m->W.data + (size_t)r * m->d;
            for (int t = 0; t < m->B; t++) m->o[(size_t)t * m->n + r] = dot_bf(w, m->x + (size_t)t * m->d, m->d);
        }
    }
}
static void mm(float *o, Dense W, const float *x, int B, int n, int d) {
    uint64_t want = W.dtype == 4 ? (uint64_t)n * d / 2 : (uint64_t)n * d * (W.dtype == 0 ? 4 : W.dtype == 3 ? 1 : 2);
    if (W.dtype == 2 || W.nbytes != want) die("matrix shape mismatch");
    double t0 = now();
    if (W.dtype == 3 || W.dtype == 4) quant_act(x, B, d);
    MM m = {o, W, x, B, n, d}; parfor(&m, mm_chunk);
    T_DENSE += now() - t0;
}

static void zc_rmsnorm(float *o, const float *x, const float *w, int n) {   /* x * rsqrt(mean x^2) * (off + w): off 1 = zero-centered (Qwen3.5/3.6), 0 = standard (Qwen3) */
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * ((float)NORM_OFF + w[i]);
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplus(float x) { return x > 20 ? x : log1pf(expf(x)); }

/* ------------------------------------------------------------------ ternary experts
 * Lookup-table math: for every 8 inputs, precompute the sum for all 256 on/off
 * patterns once. Then each ternary weight byte is one table read: exact float
 * answers, no multiplies, and one table serves all 8 experts of a layer. */

typedef struct { const uint8_t *base; size_t len; uint32_t E, nmat, rows, cols, gpr; size_t mat_bytes; } XFile;
static int FLOAT_ACT = 0;                 /* reference path: walk the set bits (verification) */

static XFile xfile(int l, const char *kind) {
    XFile x; memset(&x, 0, sizeof x);
    char path[8192]; snprintf(path, sizeof path, "%s/experts/L%02d.%s", PLATE, l, kind);
    x.base = map_file(path, &x.len);
    if (!x.base) { fprintf(stderr, "holomoe: missing %s\n", path); exit(1); }
    const uint32_t *h = (const uint32_t *)x.base;
    if (memcmp(x.base, "HMX1", 4) || h[1] != 2) die("expert file is not plate v2 (rerun shardpress with the new holopress)");
    x.E = h[2]; x.nmat = h[3]; x.rows = h[4]; x.cols = h[5]; x.gpr = h[7];
    x.mat_bytes = (size_t)x.rows * x.gpr * 34;
    return x;
}
static const uint8_t *xmat(const XFile *x, int e, int m) { return x->base + 64 + ((size_t)e * x->nmat + (size_t)m) * x->mat_bytes; }

static void lut_build(float *lut, const float *x, int cols) {      /* lut[cols/8][256] */
    for (int c = 0; c < cols / 8; c++) {
        float *t = lut + (size_t)c * 256; const float *v = x + c * 8; t[0] = 0;
        for (int p = 1; p < 256; p++) t[p] = t[p & (p - 1)] + v[__builtin_ctz(p)];
    }
}

typedef struct { float *o; const uint8_t *mat; const float *lut, *x; int rows, gpr; } TL;
static void tl_chunk(void *c, size_t k) {
    TL *t = c; int per = (t->rows + NCH - 1) / NCH, a = (int)k * per, b = a + per; if (b > t->rows) b = t->rows;
    const bf16 *sc = (const bf16 *)(t->mat + (size_t)t->rows * t->gpr * 32);
    for (int r = a; r < b; r++) {
        const uint64_t *m = (const uint64_t *)t->mat + (size_t)r * t->gpr * 4; float acc = 0;
        for (int g = 0; g < t->gpr; g++, m += 4) {
            float scale = bf2f(sc[(size_t)r * t->gpr + g]); if (scale == 0) continue;
            float s = 0;
            if (FLOAT_ACT) {
                for (int w = 0; w < 2; w++) {
                    uint64_t p = m[w], n = m[2 + w]; const float *x = t->x + g * 128 + w * 64;
                    while (p) { s += x[__builtin_ctzll(p)]; p &= p - 1; }
                    while (n) { s -= x[__builtin_ctzll(n)]; n &= n - 1; }
                }
            } else {
                const float *L = t->lut + (size_t)g * 16 * 256;
                for (int w = 0; w < 2; w++) {
                    uint64_t p = m[w], n = m[2 + w]; const float *Lw = L + (size_t)w * 8 * 256;
                    for (int q = 0; q < 8; q++) s += Lw[q * 256 + ((p >> (8 * q)) & 255)] - Lw[q * 256 + ((n >> (8 * q)) & 255)];
                }
            }
            acc += scale * s;
        }
        t->o[r] = acc;
    }
}
static void tern(float *o, const uint8_t *mat, int rows, int gpr, const float *lut, const float *x) {
    TL t = {o, mat, lut, x, rows, gpr}; parfor(&t, tl_chunk);
}

/* ------------------------------------------------------------------ model */

typedef struct {
    int full;
    float *ln1, *ln2;
    Dense qkv, z, bproj, aproj, out; float *conv_w, *A_log, *dt_bias, *gnorm;
    Dense q, k, v, o; float *qn, *kn;
    Dense router, sg, su, sd, sgate;
    Dense mg, mu, md;                       /* dense MLP (no experts, e.g. Qwen3.5-2B) */
    XFile gu, dn;
} Layer;

static Layer *LY;
static Dense EMB, LMH; static float *FNORM, *INVF;
static int CTX = 4096;
#define PB 32                             /* tokens per batch while reading a prompt */

static float **SREC, **SCONV, **KC, **VC;
#define MAXK 8                            /* most words guessed per pass */
static int SNAPON; static float **SNAPS, **SNAPC;   /* per-position copies of the linear memories [MAXK+1][NL] */
static float MINW = 0;                    /* skip experts whose share is below this */
static long N_EXP_RUN, N_EXP_SKIP, N_HIT, N_PROBE;
static long POS;
static long *HEAT;                        /* expert usage, this session [NL][NE] */
static uint64_t *HEAT_ALL;                /* lifetime usage, saved in the plate */


/* ------------------------------------------------------------------ trimmed vocabulary
 * The model knows 248k tokens across dozens of writing systems. For Latin-script text,
 * math and code we only score the tokens that can appear there (every single byte is
 * kept, so nothing becomes impossible - rarer scripts just spell out byte by byte).
 * Turned off automatically for any question written in another script. */
static int TRIM = 1, USE_TRIM = 1, TRIMN; static int *TRIMIDS; static Dense LMT; static float *LOGT; static double TRIM_SAVED;
static int cp_ok(uint32_t c) {
    return c < 0x80 || (c >= 0xA0 && c <= 0x24F) || (c >= 0x370 && c <= 0x3FF) || (c >= 0x2000 && c <= 0x2BFF) ||
           (c >= 0x1D400 && c <= 0x1D7FF) || c == 0xFEFF || (c >= 0x1F300 && c <= 0x1FAFF);
}
static int utf8_ok_run(const uint8_t *s, uint32_t n) {   /* every character complete and in an allowed range? */
    uint32_t i = 0;
    while (i < n) {
        uint8_t b = s[i]; uint32_t c, k;
        if (b < 0x80) { i++; continue; }
        if ((b >> 5) == 6) { c = b & 31; k = 1; } else if ((b >> 4) == 14) { c = b & 15; k = 2; } else if ((b >> 3) == 30) { c = b & 7; k = 3; } else return 0;
        if (i + k >= n) return 0;
        for (uint32_t j = 1; j <= k; j++) { if ((s[i + j] & 0xC0) != 0x80) return 0; c = c << 6 | (s[i + j] & 63); }
        if (!cp_ok(c)) return 0;
        i += k + 1;
    }
    return 1;
}
static int tok_keep(int id) {
    if (id < 0 || id >= NTOK) return 0;
    if (TK[id].rank == -2) return 1;                       /* special tokens */
    if (TK[id].len <= 1) return 1;                         /* every single byte stays available */
    return utf8_ok_run(TK[id].s, TK[id].len);
}
static void trim_build(void) {
    if (!TRIM || LMH.dtype != 3) { TRIM = 0; return; }
    TRIMIDS = xcalloc((size_t)VOCAB, sizeof(int)); TRIMN = 0;
    for (int i = 0; i < VOCAB; i++) if (tok_keep(i)) TRIMIDS[TRIMN++] = i;
    if (TRIMN == 0 || TRIMN > VOCAB * 9 / 10) { TRIM = 0; return; }   /* not worth it */
    size_t gpr = (size_t)DIM / 32;
    int8_t *q = xcalloc((size_t)TRIMN * DIM, 1); float *rs = xcalloc((size_t)TRIMN * gpr, 4);
    for (int k = 0; k < TRIMN; k++) {
        memcpy(q + (size_t)k * DIM, (const int8_t *)LMH.data + (size_t)TRIMIDS[k] * DIM, (size_t)DIM);
        memcpy(rs + (size_t)k * gpr, LMH.rs + (size_t)TRIMIDS[k] * gpr, gpr * 4);
    }
    LMT = LMH; LMT.data = q; LMT.rs = rs; LMT.nbytes = (uint64_t)TRIMN * DIM; LMT.shape[0] = (uint64_t)TRIMN;
    double full = (double)VOCAB * DIM * 1.125, part = (double)TRIMN * DIM * 1.125;
    TRIM_SAVED = full - part; Q8_BYTES -= TRIM_SAVED;
    long pg = sysconf(_SC_PAGESIZE); uintptr_t a = (uintptr_t)LMH.data & ~(uintptr_t)(pg - 1);
    madvise((void *)a, (size_t)VOCAB * DIM + ((uintptr_t)LMH.data - a), MADV_DONTNEED);   /* full table stays on disk until needed */
}

static void model_load(void) {
    char nm[512];
    {   /* tensor prefix: multimodal checkpoints nest the text model under model.language_model. */
        const char *cand[] = {"model.language_model.", "model."}; PFX[0] = 0;
        for (int i = 0; i < 2 && !PFX[0]; i++) { char p2[8192]; struct stat sb; snprintf(p2, sizeof p2, "%s/dense/%sembed_tokens.weight.t", PLATE, cand[i]); if (!stat(p2, &sb)) snprintf(PFX, sizeof PFX, "%s", cand[i]); }
        if (!PFX[0]) die("plate has no embed_tokens (is the download finished?)");
    }
    snprintf(nm, sizeof nm, "%sembed_tokens.weight", PFX); EMB = dense(nm, 1);
    LMH = dense("lm_head.weight", 0); if (!LMH.data) { fprintf(stderr, "[no lm_head - using tied embeddings]\n"); LMH = EMB; }
    q8ify(&LMH, "lm_head");
    trim_build();
    snprintf(nm, sizeof nm, "%snorm.weight", PFX); FNORM = to_f32(dense(nm, 1), (size_t)DIM);
    LY = xcalloc((size_t)NL, sizeof(Layer));
    int kdim = LKH * LKD, vdim = LVH * LVD, cdim = 2 * kdim + vdim;
#define P(fmt) (snprintf(nm, sizeof nm, "%slayers.%d." fmt, PFX, l), nm)
    for (int l = 0; l < NL; l++) {
        Layer *y = &LY[l]; y->full = LTYPES[l] == 'F';
        y->ln1 = to_f32(dense(P("input_layernorm.weight"), 1), (size_t)DIM);
        y->ln2 = to_f32(dense(P("post_attention_layernorm.weight"), 1), (size_t)DIM);
        if (!y->full) {
            y->qkv = dense(P("linear_attn.in_proj_qkv.weight"), 1); y->z = dense(P("linear_attn.in_proj_z.weight"), 1);
            y->bproj = dense(P("linear_attn.in_proj_b.weight"), 1); y->aproj = dense(P("linear_attn.in_proj_a.weight"), 1);
            y->out = dense(P("linear_attn.out_proj.weight"), 1);
            y->conv_w = to_f32(dense(P("linear_attn.conv1d.weight"), 1), (size_t)cdim * CONVK);
            y->A_log = to_f32(dense(P("linear_attn.A_log"), 1), (size_t)LVH);
            y->dt_bias = to_f32(dense(P("linear_attn.dt_bias"), 1), (size_t)LVH);
            y->gnorm = to_f32(dense(P("linear_attn.norm.weight"), 1), (size_t)LVD);
        } else {
            y->q = dense(P("self_attn.q_proj.weight"), 1); y->k = dense(P("self_attn.k_proj.weight"), 1);
            y->v = dense(P("self_attn.v_proj.weight"), 1); y->o = dense(P("self_attn.o_proj.weight"), 1);
            y->qn = to_f32(dense(P("self_attn.q_norm.weight"), 1), (size_t)HD);
            y->kn = to_f32(dense(P("self_attn.k_norm.weight"), 1), (size_t)HD);
        }
        if (NE) y->router = dense(P("mlp.gate.weight"), 1);
        else { y->mg = dense(P("mlp.gate_proj.weight"), 1); y->mu = dense(P("mlp.up_proj.weight"), 1); y->md = dense(P("mlp.down_proj.weight"), 1); }
        if (SHI) {
            y->sg = dense(P("mlp.shared_expert.gate_proj.weight"), 1); y->su = dense(P("mlp.shared_expert.up_proj.weight"), 1);
            y->sd = dense(P("mlp.shared_expert.down_proj.weight"), 1); y->sgate = dense(P("mlp.shared_expert_gate.weight"), 1);
        }
        Dense *qs[] = {&y->qkv, &y->z, &y->bproj, &y->aproj, &y->out, &y->q, &y->k, &y->v, &y->o, &y->sg, &y->su, &y->sd, &y->mg, &y->mu, &y->md};   /* router stays exact */
        const char *qn[] = {"qkv", "z", "b", "a", "out", "q", "k", "v", "o", "sg", "su", "sd", "mg", "mu", "md"};
        for (size_t qi = 0; qi < sizeof qs / sizeof *qs; qi++) { char nm2[64]; snprintf(nm2, sizeof nm2, "L%02d.%s", l, qn[qi]); if (LEAN && Q8) q4ify(qs[qi], nm2); else q8ify(qs[qi], nm2); }
        if (NE) y->gu = xfile(l, "gu"), y->dn = xfile(l, "dn");
        if (NE && ((int)y->gu.E != NE || (int)y->gu.rows != MOEI || (int)y->gu.cols != DIM)) die("expert file shape mismatch");
        fprintf(stderr, "\r[%s attention: layer %d/%d]", Q8_BUILT > 0 ? "building + saving int8" : "loading", l + 1, NL); fflush(stderr);
    }
#undef P
    fprintf(stderr, "\n");
    INVF = xcalloc((size_t)ROT / 2 + 1, 4);
    for (int i = 0; i < ROT / 2; i++) INVF[i] = 1.0f / powf(THETA, (float)(2 * i) / (float)ROT);
    SREC = xcalloc((size_t)NL, sizeof(float *)); SCONV = xcalloc((size_t)NL, sizeof(float *));
    KC = xcalloc((size_t)NL, sizeof(float *)); VC = xcalloc((size_t)NL, sizeof(float *));
    for (int l = 0; l < NL; l++) {
        if (!LY[l].full) { SREC[l] = xcalloc((size_t)LVH * LKD * LVD, 4); SCONV[l] = xcalloc((size_t)cdim * CONVK, 4); }
        else { KC[l] = xcalloc((size_t)CTX * NKV * HD, 4); VC[l] = xcalloc((size_t)CTX * NKV * HD, 4); }
    }
    HEAT = xcalloc((size_t)NL * NE, sizeof(long));
    HEAT_ALL = xcalloc((size_t)NL * NE, sizeof(uint64_t));
}

static void state_reset(void) {
    int cdim = 2 * LKH * LKD + LVH * LVD;
    for (int l = 0; l < NL; l++)
        if (!LY[l].full) { memset(SREC[l], 0, (size_t)LVH * LKD * LVD * 4); memset(SCONV[l], 0, (size_t)cdim * CONVK * 4); }
    POS = 0;
}

/* ------------------------------------------------------------------ heat map + preload */

static void heat_path(char *p, size_t n) { snprintf(p, n, "%s/heat.bin", PLATE); }
static void heat_load(void) {
    char p[8192]; heat_path(p, sizeof p); FILE *f = fopen(p, "rb"); if (!f) return;
    uint32_t h[3]; if (fread(h, 4, 3, f) == 3 && !memcmp(h, "HMHT", 4) && (int)h[1] == NL && (int)h[2] == NE)
        if (fread(HEAT_ALL, 8, (size_t)NL * NE, f) != (size_t)NL * NE) memset(HEAT_ALL, 0, (size_t)NL * NE * 8);
    fclose(f);
}
static void heat_save(void) {                    /* lifetime heat += this session, atomic write */
    if (!HEAT || !HEAT_ALL) return;
    long any = 0; for (long i = 0; i < (long)NL * NE; i++) any |= HEAT[i];
    if (!any) return;
    char p[8192], tp[8300]; heat_path(p, sizeof p); snprintf(tp, sizeof tp, "%s.part", p);
    FILE *f = fopen(tp, "wb"); if (!f) return;
    uint32_t h[3] = {0, (uint32_t)NL, (uint32_t)NE}; memcpy(h, "HMHT", 4);
    uint64_t *a = xcalloc((size_t)NL * NE, 8);
    for (long i = 0; i < (long)NL * NE; i++) a[i] = HEAT_ALL[i] + (uint64_t)HEAT[i];
    int ok = fwrite(h, 4, 3, f) == 3 && fwrite(a, 8, (size_t)NL * NE, f) == (size_t)NL * NE;
    ok = fflush(f) == 0 && fsync(fileno(f)) == 0 && ok; fclose(f);
    if (ok && rename(tp, p) == 0) { memcpy(HEAT_ALL, a, (size_t)NL * NE * 8); memset(HEAT, 0, (size_t)NL * NE * sizeof(long)); }
    free(a);
}

static double phys_ram(void) {
#if defined(__APPLE__)
    uint64_t m = 0; size_t l = sizeof m; if (sysctlbyname("hw.memsize", &m, &l, NULL, 0) == 0) return (double)m;
    return 8e9;
#else
    return (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE);
#endif
}

static double PRELOAD_GB = -1; static long PRELOADED; static double PRELOAD_BYTES;
typedef struct { uint64_t heat; int l, e; } Rank;
static int rank_cb(const void *a, const void *b) {
    const Rank *x = a, *y = b;
    if (x->heat != y->heat) return x->heat < y->heat ? 1 : -1;
    return x->e != y->e ? x->e - y->e : x->l - y->l;
}
static void preload(void) {
    if (!NE) return;                                     /* dense model: nothing to stream */
    double per = (double)(LY[0].gu.mat_bytes * 2 + LY[0].dn.mat_bytes);
    double budget = PRELOAD_GB >= 0 ? PRELOAD_GB * 1e9 : phys_ram() - Q8_BYTES - 3.0e9;   /* leave ~3 GB for macOS and apps */
    long want = budget > 0 ? (long)(budget / per) : 0;
    if (want > (long)NL * NE) want = (long)NL * NE;
    if (want <= 0) return;
    Rank *r = xcalloc((size_t)NL * NE, sizeof(Rank));
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) r[l * NE + e] = (Rank){HEAT_ALL[(size_t)l * NE + e], l, e};
    qsort(r, (size_t)NL * NE, sizeof(Rank), rank_cb);
    double t0 = now();
    for (long i = 0; i < want; i++) {
        const Layer *y = &LY[r[i].l];
        touch(xmat(&y->gu, r[i].e, 0), y->gu.mat_bytes * 2);
        touch(xmat(&y->dn, r[i].e, 0), y->dn.mat_bytes);
        PRELOADED++; PRELOAD_BYTES += per;
        if (i % 64 == 0 || i == want - 1) { fprintf(stderr, "\r[warming experts into RAM: %ld/%ld  %.2f GB]", i + 1, want, PRELOAD_BYTES / 1e9); fflush(stderr); }
    }
    fprintf(stderr, "  %.1fs\n", now() - t0);
    free(r);
}

/* ------------------------------------------------------------------ batched forward */

static int PREDICT = 1; static float *PRB, *PRL; static int PIDX[PB][64], PRED_L = -1; static long N_PRED, N_PRED_HIT;     /* next-layer expert prediction scratch */
static float *LOGB, *XB, *HB, *OB, *QKVB, *ZB, *BAB, *AOB, *KB, *VB, *GTB, *ATT, *LOG, *LUTB, *LUTD, *EG, *EU, *EY;
static int BIG;

static void scratch_alloc(void) {
    int kdim = LKH * LKD, vdim = LVH * LVD, cdim = 2 * kdim + vdim;
    BIG = cdim; if (NH * HD * 2 > BIG) BIG = NH * HD * 2; if (NE > BIG) BIG = NE;
    if (DIM > BIG) BIG = DIM;
    if (SHI > BIG) BIG = SHI;
    if (MOEI > BIG) BIG = MOEI;
    if (DENSE_I > BIG) BIG = DENSE_I;
    XB = xcalloc((size_t)PB * DIM, 4); HB = xcalloc((size_t)PB * DIM, 4); OB = xcalloc((size_t)PB * DIM, 4);
    QKVB = xcalloc((size_t)PB * BIG, 4); ZB = xcalloc((size_t)PB * BIG, 4); BAB = xcalloc((size_t)PB * BIG, 4);
    AOB = xcalloc((size_t)PB * BIG, 4); KB = xcalloc((size_t)PB * BIG, 4); VB = xcalloc((size_t)PB * BIG, 4);
    GTB = xcalloc((size_t)PB * BIG, 4);
    LOG = xcalloc((size_t)VOCAB, 4); ATT = xcalloc((size_t)CTX, 4);
    LOGB = xcalloc((size_t)PB * VOCAB, 4);
    LUTB = xcalloc((size_t)PB * (DIM / 8) * 256, 4);
    LUTD = xcalloc((size_t)(MOEI / 8) * 256, 4);
    EG = xcalloc((size_t)MOEI, 4); EU = xcalloc((size_t)MOEI, 4); EY = xcalloc((size_t)DIM, 4);
    if (LVD > 512 || HD > 1024) die("head size larger than supported");
}

static void linear_attn(int l, int B) {
    Layer *y = &LY[l]; int kdim = LKH * LKD, vdim = LVH * LVD, cdim = 2 * kdim + vdim;
    mm(QKVB, y->qkv, HB, B, cdim, DIM); mm(ZB, y->z, HB, B, vdim, DIM);
    mm(BAB, y->bproj, HB, B, LVH, DIM); mm(GTB, y->aproj, HB, B, LVH, DIM);
    double t0 = now();
    for (int t = 0; t < B; t++) {
        float *qkv = QKVB + (size_t)t * cdim, *z = ZB + (size_t)t * vdim, *bb = BAB + (size_t)t * LVH, *aa = GTB + (size_t)t * LVH, *o = AOB + (size_t)t * vdim;
        float *cs = SCONV[l];                                  /* causal depthwise conv over time */
        for (int c = 0; c < cdim; c++) {
            float *st = cs + (size_t)c * CONVK; const float *w = y->conv_w + (size_t)c * CONVK;
            for (int j = 0; j < CONVK - 1; j++) st[j] = st[j + 1];
            st[CONVK - 1] = qkv[c];
            float acc = 0; for (int j = 0; j < CONVK; j++) acc += w[j] * st[j];
            qkv[c] = silu(acc);
        }
        float *q = qkv, *k = qkv + kdim, *v = qkv + 2 * kdim;
        for (int hh = 0; hh < LKH; hh++) {                     /* l2-normalize q and k per head */
            float *qq = q + hh * LKD, *kk = k + hh * LKD; double sq = 0, sk = 0;
            for (int i = 0; i < LKD; i++) { sq += (double)qq[i] * qq[i]; sk += (double)kk[i] * kk[i]; }
            float rq = 1.0f / sqrtf((float)sq + 1e-6f) / sqrtf((float)LKD), rk = 1.0f / sqrtf((float)sk + 1e-6f);
            for (int i = 0; i < LKD; i++) { qq[i] *= rq; kk[i] *= rk; }
        }
        int ratio = LVH / LKH;
        for (int j = 0; j < LVH; j++) {                        /* gated delta rule */
            const float *qq = q + (j / ratio) * LKD, *kk = k + (j / ratio) * LKD, *vv = v + j * LVD;
            float beta = sigm(bb[j]), g = -expf(y->A_log[j]) * softplus(aa[j] + y->dt_bias[j]), dec = expf(g);
            float *S = SREC[l] + (size_t)j * LKD * LVD, *oo = o + j * LVD, kvm[512], del[512];
            for (int c = 0; c < LVD; c++) kvm[c] = 0;
            for (int r = 0; r < LKD; r++) { float kr = kk[r]; float *row = S + (size_t)r * LVD; for (int c = 0; c < LVD; c++) { row[c] *= dec; kvm[c] += row[c] * kr; } }
            for (int c = 0; c < LVD; c++) { del[c] = (vv[c] - kvm[c]) * beta; oo[c] = 0; }
            for (int r = 0; r < LKD; r++) {
                float kr = kk[r], qr = qq[r]; float *row = S + (size_t)r * LVD;
                for (int c = 0; c < LVD; c++) { row[c] += kr * del[c]; oo[c] += row[c] * qr; }
            }
            double ss = 0; for (int c = 0; c < LVD; c++) ss += (double)oo[c] * oo[c];
            float rr = 1.0f / sqrtf((float)(ss / LVD) + EPS);
            for (int c = 0; c < LVD; c++) oo[c] = oo[c] * rr * y->gnorm[c] * silu(z[j * LVD + c]);
        }
        if (SNAPON) {
            memcpy(SNAPS[(size_t)t * NL + l], SREC[l], (size_t)LVH * LKD * LVD * 4);
            memcpy(SNAPC[(size_t)t * NL + l], SCONV[l], (size_t)cdim * CONVK * 4);
        }
    }
    T_ATT += now() - t0;
    mm(OB, y->out, AOB, B, DIM, vdim);
}

static void full_attn(int l, int B) {
    Layer *y = &LY[l]; int qm = ATTN_GATE ? 2 : 1, qd = NH * HD * qm, kv = NKV * HD, half = ROT / 2, grp = NH / NKV;
    mm(QKVB, y->q, HB, B, qd, DIM); mm(KB, y->k, HB, B, kv, DIM); mm(VB, y->v, HB, B, kv, DIM);
    double t0 = now();
    for (int t = 0; t < B; t++) {
        long pos = POS + t; int slot = (int)(pos % CTX), n = pos + 1 < CTX ? (int)pos + 1 : CTX;
        float *qg = QKVB + (size_t)t * qd, *kk = KB + (size_t)t * kv, *vv = VB + (size_t)t * kv;
        float *q = ZB + (size_t)t * NH * HD, *gate = GTB + (size_t)t * NH * HD, *res = AOB + (size_t)t * NH * HD;
        for (int hh = 0; hh < NH; hh++) {                      /* split query / gate (gated families), q_norm */
            if (ATTN_GATE) memcpy(gate + hh * HD, qg + hh * HD * 2 + HD, HD * 4);
            zc_rmsnorm(q + hh * HD, qg + hh * HD * qm, y->qn, HD);
        }
        for (int hh = 0; hh < NKV; hh++) zc_rmsnorm(kk + hh * HD, kk + hh * HD, y->kn, HD);
        for (int i = 0; i < half; i++) {                       /* partial rotary positions */
            float a = (float)pos * INVF[i], c = cosf(a), s = sinf(a);
            for (int hh = 0; hh < NH; hh++) { float *p = q + hh * HD, x0 = p[i], x1 = p[i + half]; p[i] = x0 * c - x1 * s; p[i + half] = x1 * c + x0 * s; }
            for (int hh = 0; hh < NKV; hh++) { float *p = kk + hh * HD, x0 = p[i], x1 = p[i + half]; p[i] = x0 * c - x1 * s; p[i + half] = x1 * c + x0 * s; }
        }
        memcpy(KC[l] + (size_t)slot * kv, kk, (size_t)kv * 4);
        memcpy(VC[l] + (size_t)slot * kv, vv, (size_t)kv * 4);
        float scale = 1.0f / sqrtf((float)HD);
        for (int hh = 0; hh < NH; hh++) {
            const float *qq = q + hh * HD; int ko = (hh / grp) * HD; float mx = -1e30f, sum = 0;
            for (int s2 = 0; s2 < n; s2++) { const float *kr = KC[l] + (size_t)s2 * kv + ko; float d = 0; for (int i = 0; i < HD; i++) d += qq[i] * kr[i]; ATT[s2] = d * scale; if (ATT[s2] > mx) mx = ATT[s2]; }
            for (int s2 = 0; s2 < n; s2++) { ATT[s2] = expf(ATT[s2] - mx); sum += ATT[s2]; }
            float *rr = res + hh * HD; for (int i = 0; i < HD; i++) rr[i] = 0;
            for (int s2 = 0; s2 < n; s2++) { const float *vr = VC[l] + (size_t)s2 * kv + ko; float p = ATT[s2] / sum; for (int i = 0; i < HD; i++) rr[i] += p * vr[i]; }
        }
        if (ATTN_GATE) for (int i = 0; i < NH * HD; i++) res[i] *= sigm(gate[i]);
    }
    T_ATT += now() - t0;
    mm(OB, y->o, AOB, B, DIM, NH * HD);
}

static int resident(const uint8_t *p) {                 /* is this expert already in RAM? */
    long pg = sysconf(_SC_PAGESIZE); unsigned char v = 0;
    void *a = (void *)((uintptr_t)p & ~(uintptr_t)(pg - 1));
    if (mincore(a, (size_t)pg, (void *)&v) != 0) return 0;
    return v & 1;
}
static void prefetch_expert(const XFile *x, int e) {
    long pg = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t)xmat(x, e, 0), b = a + x->mat_bytes * x->nmat;
    a &= ~(uintptr_t)(pg - 1);
    madvise((void *)a, b - a, MADV_WILLNEED);
}

static void dense_mlp(int l, int B) {                 /* SwiGLU MLP, int8 like attention */
    Layer *y = &LY[l]; float *g = KB, *u = VB;
    mm(g, y->mg, HB, B, DENSE_I, DIM); mm(u, y->mu, HB, B, DENSE_I, DIM);
    for (size_t i = 0; i < (size_t)B * DENSE_I; i++) g[i] = silu(g[i]) * u[i];
    mm(OB, y->md, g, B, DIM, DENSE_I);
}
static void moe(int l, int B) {
    if (!NE) { dense_mlp(l, B); return; }
    Layer *y = &LY[l]; float *lg = QKVB;                       /* router logits [B][NE] */
    mm(lg, y->router, HB, B, NE, DIM);
    double t0 = now();
    int idx[PB][64]; float w[PB][64];
    for (int t = 0; t < B; t++) {
        float *p = lg + (size_t)t * NE, mx = p[0]; for (int e = 1; e < NE; e++) if (p[e] > mx) mx = p[e];
        double z = 0; for (int e = 0; e < NE; e++) { p[e] = expf(p[e] - mx); z += p[e]; }
        float ws = 0;
        for (int k = 0; k < TOPK; k++) {
            int best = -1; for (int e = 0; e < NE; e++) { int used = 0; for (int j = 0; j < k; j++) if (idx[t][j] == e) used = 1; if (!used && (best < 0 || p[e] > p[best])) best = e; }
            idx[t][k] = best; w[t][k] = (float)(p[best] / z); ws += w[t][k];
        }
        if (NORM_TOPK) for (int k = 0; k < TOPK; k++) w[t][k] /= ws;
        if (PRED_L == l) for (int k = 0; k < TOPK; k++) { N_PRED++; for (int j = 0; j < TOPK; j++) if (PIDX[t][j] == idx[t][k]) { N_PRED_HIT++; break; } }
        if (MINW > 0) {                                        /* drop the faint ones (keep at least 2), renormalize */
            float s2 = 0;
            for (int k = 0; k < TOPK; k++) { if (k >= 2 && w[t][k] < MINW) { w[t][k] = 0; N_EXP_SKIP++; } s2 += w[t][k]; }
            if (NORM_TOPK) for (int k = 0; k < TOPK; k++) w[t][k] /= s2;
        }
        for (int k = 0; k < TOPK; k++) {
            if (w[t][k] == 0) continue;
            HEAT[(size_t)l * NE + idx[t][k]]++; N_EXP_RUN++;
            if (resident(xmat(&y->gu, idx[t][k], 0))) N_HIT++;
            N_PROBE++;
            prefetch_expert(&y->gu, idx[t][k]); prefetch_expert(&y->dn, idx[t][k]);
        }
        if (!FLOAT_ACT) lut_build(LUTB + (size_t)t * (DIM / 8) * 256, HB + (size_t)t * DIM, DIM);
    }
    memset(OB, 0, (size_t)B * DIM * 4);
    for (int e = 0; e < NE; e++)                               /* expert by expert: each one is read once per batch */
        for (int t = 0; t < B; t++)
            for (int k = 0; k < TOPK; k++) {
                if (idx[t][k] != e || w[t][k] == 0) continue;
                const float *lut = LUTB + (size_t)t * (DIM / 8) * 256, *h = HB + (size_t)t * DIM;
                tern(EG, xmat(&y->gu, e, 0), MOEI, y->gu.gpr, lut, h);
                tern(EU, xmat(&y->gu, e, 1), MOEI, y->gu.gpr, lut, h);
                for (int i = 0; i < MOEI; i++) EG[i] = silu(EG[i]) * EU[i];
                if (!FLOAT_ACT) lut_build(LUTD, EG, MOEI);
                tern(EY, xmat(&y->dn, e, 0), DIM, y->dn.gpr, LUTD, EG);
                float *o = OB + (size_t)t * DIM; for (int i = 0; i < DIM; i++) o[i] += w[t][k] * EY[i];
            }
    T_EXP += now() - t0;
    if (SHI) {
        float *g = KB, *u = VB, *yo = AOB, sgv[PB];
        mm(g, y->sg, HB, B, SHI, DIM); mm(u, y->su, HB, B, SHI, DIM);
        for (size_t i = 0; i < (size_t)B * SHI; i++) g[i] = silu(g[i]) * u[i];
        mm(yo, y->sd, g, B, DIM, SHI);
        mm(sgv, y->sgate, HB, B, 1, DIM);
        for (int t = 0; t < B; t++) { float s = sigm(sgv[t]); for (int i = 0; i < DIM; i++) OB[(size_t)t * DIM + i] += s * yo[(size_t)t * DIM + i]; }
    }
}

/* run n tokens through the model; logits for the last one if want */
static void head_logits(float *dst, const float *h, int B) {
    if (USE_TRIM && TRIM && TRIMN) {
        if (!LOGT) LOGT = xcalloc((size_t)PB * TRIMN, 4);
        mm(LOGT, LMT, h, B, TRIMN, DIM);
        for (int t = 0; t < B; t++) {
            float *row = dst + (size_t)t * VOCAB; const float *lt = LOGT + (size_t)t * TRIMN;
            for (int i = 0; i < VOCAB; i++) row[i] = -1e30f;
            for (int k = 0; k < TRIMN; k++) row[TRIMIDS[k]] = lt[k];
        }
    } else mm(dst, LMH, h, B, VOCAB, DIM);
}
static void forward_batch(const int *ids, int n, int want) {
    for (int s = 0; s < n; s += PB) {
        int B = n - s < PB ? n - s : PB;
        for (int t = 0; t < B; t++) {
            int tok = ids[s + t]; float *x = XB + (size_t)t * DIM;
            for (int i = 0; i < DIM; i++) x[i] = EMB.dtype == 0 ? ((const float *)EMB.data)[(size_t)tok * DIM + i] : bf2f(((const bf16 *)EMB.data)[(size_t)tok * DIM + i]);
        }
        for (int l = 0; l < NL; l++) {
            if (PREDICT && NE && l + 1 < NL) {                       /* ask layer l+1's router now, so the SSD fetches while we compute */
                if (!PRB) { PRB = xcalloc((size_t)PB * DIM, 4); PRL = xcalloc((size_t)PB * NE, 4); }
                for (int t = 0; t < B; t++) zc_rmsnorm(PRB + (size_t)t * DIM, XB + (size_t)t * DIM, LY[l + 1].ln2, DIM);
                mm(PRL, LY[l + 1].router, PRB, B, NE, DIM);
                for (int t = 0; t < B; t++) {
                    const float *p = PRL + (size_t)t * NE; int used[64];
                    for (int k = 0; k < TOPK; k++) {
                        int b = -1; for (int e = 0; e < NE; e++) { int u = 0; for (int j = 0; j < k; j++) if (used[j] == e) u = 1; if (!u && (b < 0 || p[e] > p[b])) b = e; }
                        used[k] = b; PIDX[t][k] = b; prefetch_expert(&LY[l + 1].gu, b); prefetch_expert(&LY[l + 1].dn, b);
                    }
                }
                PRED_L = l + 1;
            }
            for (int t = 0; t < B; t++) zc_rmsnorm(HB + (size_t)t * DIM, XB + (size_t)t * DIM, LY[l].ln1, DIM);
            if (LY[l].full) full_attn(l, B); else linear_attn(l, B);
            for (size_t i = 0; i < (size_t)B * DIM; i++) XB[i] += OB[i];
            for (int t = 0; t < B; t++) zc_rmsnorm(HB + (size_t)t * DIM, XB + (size_t)t * DIM, LY[l].ln2, DIM);
            moe(l, B);
            for (size_t i = 0; i < (size_t)B * DIM; i++) XB[i] += OB[i];
        }
        POS += B;
        if (want == 2) {
            if (n > PB) die("internal: all-logits batch larger than PB");                                       /* logits for every position (guess checking, quality test) */
            double t0 = now();
            for (int t = 0; t < B; t++) zc_rmsnorm(HB + (size_t)t * DIM, XB + (size_t)t * DIM, FNORM, DIM);
            double td = T_DENSE; head_logits(LOGB + (size_t)s * VOCAB, HB, B); T_DENSE = td;
            T_HEAD += now() - t0;
        } else if (want && s + B == n) {
            double t0 = now();
            zc_rmsnorm(HB, XB + (size_t)(B - 1) * DIM, FNORM, DIM);
            double td = T_DENSE; head_logits(LOG, HB, 1); T_DENSE = td;
            T_HEAD += now() - t0;
        }
    }
}
static void forward(int tok, int want) { forward_batch(&tok, 1, want); }

/* ------------------------------------------------------------------ terminal renderer
 * The model writes Markdown and LaTeX. This turns it into something a terminal
 * shows well, while it streams: **bold**, `code`, # headings, - bullets, and math
 * ($...$, \( \), \[ \]) with Greek letters, symbols, fractions, x^2 and x_0. */

static int COLOR = 1, RENDER = 1;
#define C_RESET "\033[0m"
enum { G_PLAIN, G_FRAC1, G_FRAC2, G_SQRT, G_BB };
static struct {
    int bold, ital, code, block, math, dmath, head, bol, think;
    char run_c; int run_n;                     /* pending run of * $ ` # - */
    int incmd; char cmd[40]; int cl;           /* collecting \command */
    int stk[64], sp, frac_next, arg_next;
    int supsub; char grp[64]; int gl, gdepth;  /* ^{...} / _{...} buffer: supsub 1 = sup, 2 = sub; gdepth > 0 = in group */
    int pend_ss;                               /* saw ^ or _ , waiting for next char */
    int after_frac, eat_space;                 /* 1/2·m ; LaTeX ignores the space after \psi etc. */
} R;

/* all renderer output goes through here, so fractions can be captured and tidied */
static char CAPB[6][1024]; static int CAPN[6], CAPD;
static void ro_s(const char *t) {
    if (CAPD > 0) { int d = CAPD - 1; size_t n = strlen(t); if (CAPN[d] + (int)n < 1023) { memcpy(CAPB[d] + CAPN[d], t, n); CAPN[d] += (int)n; } return; }
    fputs(t, stdout);
}
static void ro_c(int c) { char b[2] = {(char)c, 0}; ro_s(b); }
static void rs_style(void) {
    if (!COLOR || CAPD) return;
    fputs(C_RESET, stdout);
    if (R.think) { fputs("\033[2;3m", stdout); return; }
    if (R.head) fputs("\033[1;35m", stdout);
    if (R.bold) fputs("\033[1m", stdout);
    if (R.ital) fputs("\033[3m", stdout);
    if (R.code || R.block) fputs("\033[36m", stdout);
    if (R.math || R.dmath) fputs("\033[33m", stdout);
}
static void rs_begin(int think) { memset(&R, 0, sizeof R); R.bol = 1; R.think = think; rs_style(); }

static const char *SUP_FROM = "0123456789+-=()niTx", *SUB_FROM = "0123456789+-=()aeoxihklmnpst";
static const char *SUP_TO[] = {"⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹","⁺","⁻","⁼","⁽","⁾","ⁿ","ⁱ","ᵀ","ˣ"};
static const char *SUB_TO[] = {"₀","₁","₂","₃","₄","₅","₆","₇","₈","₉","₊","₋","₌","₍","₎","ₐ","ₑ","ₒ","ₓ","ᵢ","ₕ","ₖ","ₗ","ₘ","ₙ","ₚ","ₛ","ₜ"};
static const char *ss_map(int sup, char c) {
    const char *f = sup ? SUP_FROM : SUB_FROM; const char *p = c ? strchr(f, c) : NULL;
    return p ? (sup ? SUP_TO : SUB_TO)[p - f] : NULL;
}
static void rs_char(int c); static void rs_cmd_done(void);
static void ss_emit(int sup, const char *s, int n) {   /* all mappable -> pretty, else ^(..) / _(..) */
    int ok = n > 0;
    for (int i = 0; i < n; i++) if (!ss_map(sup, s[i])) ok = 0;
    if (ok) { for (int i = 0; i < n; i++) ro_s(ss_map(sup, s[i])); return; }
    if (sup) ro_s(n == 1 ? "^" : "^(");
    static int depth = 0;                         /* render the group's own commands (e.g. _{\mu\nu} -> μν) */
    if (n < 0) n = 0;
    if (n > 63) n = 63;
    char tmp[64]; memcpy(tmp, s, (size_t)n); tmp[n] = 0;
    if (depth < 2) { depth++; for (int i = 0; i < n; i++) rs_char((unsigned char)tmp[i]); if (R.incmd) rs_cmd_done(); R.eat_space = 0; depth--; }
    else ro_s(tmp);
    if (sup && n > 1) ro_c(')');
}

static const char *SYM[][2] = {
    {"alpha","α"},{"beta","β"},{"gamma","γ"},{"delta","δ"},{"epsilon","ε"},{"varepsilon","ε"},{"zeta","ζ"},{"eta","η"},
    {"theta","θ"},{"vartheta","ϑ"},{"iota","ι"},{"kappa","κ"},{"lambda","λ"},{"mu","μ"},{"nu","ν"},{"xi","ξ"},{"pi","π"},
    {"rho","ρ"},{"sigma","σ"},{"tau","τ"},{"upsilon","υ"},{"phi","φ"},{"varphi","φ"},{"chi","χ"},{"psi","ψ"},{"omega","ω"},
    {"Gamma","Γ"},{"Delta","Δ"},{"Theta","Θ"},{"Lambda","Λ"},{"Xi","Ξ"},{"Pi","Π"},{"Sigma","Σ"},{"Phi","Φ"},{"Psi","Ψ"},{"Omega","Ω"},
    {"cdot","·"},{"times","×"},{"div","÷"},{"pm","±"},{"mp","∓"},{"le","≤"},{"leq","≤"},{"ge","≥"},{"geq","≥"},{"neq","≠"},{"ne","≠"},
    {"approx","≈"},{"sim","∼"},{"simeq","≃"},{"equiv","≡"},{"propto","∝"},{"infty","∞"},{"partial","∂"},{"nabla","∇"},{"hbar","ℏ"},
    {"ell","ℓ"},{"sum","Σ"},{"prod","Π"},{"int","∫"},{"iint","∬"},{"oint","∮"},{"to","→"},{"rightarrow","→"},{"Rightarrow","⇒"},
    {"leftarrow","←"},{"Leftarrow","⇐"},{"leftrightarrow","↔"},{"Leftrightarrow","⇔"},{"mapsto","↦"},{"langle","⟨"},{"rangle","⟩"},
    {"ldots","…"},{"cdots","⋯"},{"dots","…"},{"circ","∘"},{"degree","°"},{"in","∈"},{"notin","∉"},{"subset","⊂"},{"cup","∪"},{"cap","∩"},
    {"forall","∀"},{"exists","∃"},{"neg","¬"},{"wedge","∧"},{"vee","∨"},{"oplus","⊕"},{"otimes","⊗"},{"dagger","†"},{"perp","⊥"},
    {"parallel","∥"},{"angle","∠"},{"quad","  "},{"qquad","    "},{"star","⋆"},{"ast","∗"},{"prime","′"},{"emptyset","∅"},
    {"Box","□"},{"square","□"},{"therefore","∴"},{"because","∵"},{"lfloor","⌊"},{"rfloor","⌋"},{"lceil","⌈"},{"rceil","⌉"},
    {"vert","|"},{"mid","|"},{"|","‖"},{"{","{"},{"}","}"},{"$","$"},{"%","%"},{"_","_"},{"&","&"},{"#","#"},{",", " "},{";"," "},{":"," "},{"!",""},
    {"left",""},{"right",""},{"big",""},{"Big",""},{"bigg",""},{"Bigg",""},{"displaystyle",""},{"limits",""},{"nolimits",""},
    {"log","log"},{"ln","ln"},{"exp","exp"},{"sin","sin"},{"cos","cos"},{"tan","tan"},{"lim","lim"},{"max","max"},{"min","min"},{"det","det"},
};
static void rs_cmd_done(void) {                   /* a \command finished */
    R.incmd = 0; R.cmd[R.cl] = 0;
    const char *c = R.cmd;
    if (!R.math && !R.dmath) {                    /* outside math only \( \[ matter */
        if (!strcmp(c, "(")) { R.math = 1; rs_style(); return; }
        if (!strcmp(c, "[")) { R.dmath = 1; rs_style(); return; }
        ro_c('\\'); ro_s(c); return;
    }
    if (!strcmp(c, ")") || !strcmp(c, "]")) { R.math = R.dmath = 0; rs_style(); return; }
    if (!strcmp(c, "\\")) { ro_c('\n'); return; }
    if (!strcmp(c, "frac") || !strcmp(c, "dfrac") || !strcmp(c, "tfrac")) { R.frac_next = 1; return; }
    if (!strcmp(c, "sqrt")) { ro_s("√"); R.arg_next = G_SQRT; return; }
    if (!strcmp(c, "mathbb")) { R.arg_next = G_BB; return; }
    if (!strcmp(c, "text") || !strcmp(c, "mathrm") || !strcmp(c, "mathbf") || !strcmp(c, "mathit") || !strcmp(c, "operatorname") ||
        !strcmp(c, "boldsymbol") || !strcmp(c, "vec") || !strcmp(c, "hat") || !strcmp(c, "bar") || !strcmp(c, "tilde") ||
        !strcmp(c, "dot") || !strcmp(c, "ddot") || !strcmp(c, "mathcal") || !strcmp(c, "textbf") || !strcmp(c, "overline")) { R.arg_next = G_PLAIN; return; }
    for (size_t i = 0; i < sizeof SYM / sizeof *SYM; i++) if (!strcmp(c, SYM[i][0])) {
        ro_s(SYM[i][1]);
        if (i < 36 || !strcmp(c, "partial") || !strcmp(c, "hbar") || !strcmp(c, "ell")) R.eat_space = 1;   /* letters: a\psi -> aψ */
        return;
    }
    ro_s(c);                             /* unknown: show the name without the backslash */
}

static void rs_char(int c);
static void rs_flush_run(int next) {             /* decide what a run of * $ ` # - meant, now that we see the next char */
    int n = R.run_n; char rc = R.run_c; R.run_n = 0; R.run_c = 0;
    if (!n) return;
    int inm = R.math || R.dmath;
    switch (rc) {
    case '*':
        if (inm) { for (int i = 0; i < n; i++) ro_s("·"); break; }
        if (R.code || R.block) { for (int i = 0; i < n; i++) ro_c('*'); break; }
        if (n >= 2) { R.bold = !R.bold; rs_style(); if (n == 3) { R.ital = !R.ital; rs_style(); } break; }
        if (R.bol && next == ' ') { ro_s("•"); R.bol = 0; break; }
        if (next == ' ' || next == '\n' || next == 0) { if (R.ital) { R.ital = 0; rs_style(); } else ro_c('*'); break; }
        R.ital = !R.ital; rs_style(); break;
    case '$':
        if (R.code || R.block) { for (int i = 0; i < n; i++) ro_c('$'); break; }
        if (n >= 2) { R.dmath = !R.dmath; R.math = 0; rs_style(); break; }
        if (R.math) { R.math = 0; rs_style(); break; }
        if (next >= '0' && next <= '9') { ro_c('$'); break; }        /* money, not math */
        if (next == ' ' || next == '\n' || next == 0) { ro_c('$'); break; }
        R.math = 1; rs_style(); break;
    case '`':
        if (n >= 3) { R.block = !R.block; rs_style(); break; }
        if (R.block) { for (int i = 0; i < n; i++) ro_c('`'); break; }
        R.code = !R.code; rs_style(); break;
    case '#':
        if (next == ' ') { R.head = 1; rs_style(); R.bol = 0; R.run_c = 'S'; break; }  /* 'S' = swallow one space */
        for (int i = 0; i < n; i++) ro_c('#');
        break;
    case '-':
        if (n == 1 && next == ' ') { ro_s("•"); R.bol = 0; break; }
        for (int i = 0; i < n; i++) ro_c('-');
        break;
    }
}

static void rs_math_out(int c) {                  /* one character inside math */
    if (R.eat_space) { R.eat_space = 0; if (c == ' ' && R.gdepth == 0 && !R.pend_ss) return; }
    if (R.after_frac) { R.after_frac = 0; if (isalnum(c) && R.gdepth == 0 && !R.pend_ss) ro_s("·"); }
    if (R.pend_ss) {                              /* right after ^ or _ */
        int sup = R.pend_ss == 1; R.pend_ss = 0;
        if (c == '{') { R.supsub = sup ? 1 : 2; R.gdepth = 1; R.gl = 0; return; }
        char ch = (char)c; ss_emit(sup, &ch, 1); return;
    }
    if (R.gdepth > 0) {                           /* inside ^{...} or _{...}: buffer, then decide */
        if (c == '{') R.gdepth++;
        if (c == '}' && --R.gdepth == 0) { ss_emit(R.supsub == 1, R.grp, R.gl); R.supsub = 0; return; }
        if (R.gl < (int)sizeof R.grp) R.grp[R.gl++] = (char)c;
        return;
    }
    if (c == '^') { R.pend_ss = 1; return; }
    if (c == '_') { R.pend_ss = 2; return; }
    if (c == '{') {
        int t = G_PLAIN;
        if (R.frac_next) { t = R.frac_next == 1 ? G_FRAC1 : G_FRAC2; R.frac_next = 0; if (CAPD < 6) { CAPN[CAPD] = 0; CAPD++; } }
        else if (R.arg_next == G_SQRT) { t = G_SQRT; ro_c('('); }
        else if (R.arg_next == G_BB) t = G_BB;
        R.arg_next = 0;
        if (R.sp < 64) R.stk[R.sp++] = t;
        return;
    }
    if (c == '}') {
        int t = R.sp ? R.stk[--R.sp] : G_PLAIN;
        if (t == G_FRAC1) { R.frac_next = 2; return; }      /* numerator stays captured until the denominator closes */
        if (t == G_FRAC2 && CAPD >= 2) {
            char num[1024], den[1024];
            CAPD--; memcpy(den, CAPB[CAPD], (size_t)CAPN[CAPD]); den[CAPN[CAPD]] = 0;
            CAPD--; memcpy(num, CAPB[CAPD], (size_t)CAPN[CAPD]); num[CAPN[CAPD]] = 0;
            int sn = strpbrk(num, " +-=/") == NULL, sd = strpbrk(den, " +-=/*") == NULL;
            if (!sn) ro_c('(');
            ro_s(num);
            if (!sn) ro_c(')');
            ro_c('/');
            if (!sd) ro_c('(');
            ro_s(den);
            if (!sd) ro_c(')');
            R.after_frac = 1;
            return;
        }
        if (t == G_SQRT) ro_c(')');
        return;
    }
    if (R.sp && R.stk[R.sp - 1] == G_BB) {
        const char *bb = c == 'R' ? "ℝ" : c == 'N' ? "ℕ" : c == 'Z' ? "ℤ" : c == 'Q' ? "ℚ" : c == 'C' ? "ℂ" : NULL;
        if (bb) { ro_s(bb); return; }
    }
    if (R.frac_next == 2 && c != ' ') R.frac_next = 0;
    ro_c(c);
}

static void rs_char(int c) {
    if (R.gdepth > 0 && !R.incmd) { rs_math_out(c); return; }
    if (R.incmd) {
        if (isalpha(c) && R.cl < (int)sizeof R.cmd - 1) { R.cmd[R.cl++] = (char)c; return; }
        if (R.cl == 0 && c > 0 && c < 0x80) { R.cmd[R.cl++] = (char)c; rs_cmd_done(); return; }   /* \( \[ \, \{ ... */
        rs_cmd_done();
    }
    if (R.run_c == 'S') { R.run_c = 0; if (c == ' ') return; }
    int inm = R.math || R.dmath;
    int runnable = c == '*' || c == '$' || c == '`' || ((c == '#' || c == '-') && R.bol && !inm && !R.block);
    if (runnable && !(R.block && c != '`')) {
        if (R.run_n && R.run_c == c) { R.run_n++; return; }
        rs_flush_run(c);
        R.run_c = (char)c; R.run_n = 1; return;
    }
    if (R.run_n) rs_flush_run(c);
    if (R.run_c == 'S') { R.run_c = 0; if (c == ' ') return; }
    if (c == '\\' && !R.code && !R.block) { R.incmd = 1; R.cl = 0; return; }
    if (c == '\n') {
        if (R.head) { R.head = 0; rs_style(); }
        R.pend_ss = 0; R.bol = 1; ro_c('\n'); return;
    }
    if (c != ' ' && c != '\t') R.bol = 0;
    if ((R.math || R.dmath) && !R.code) { rs_math_out(c); return; }
    R.after_frac = 0; R.eat_space = 0;
    ro_c(c);
}
static void rs_end(void) {                        /* end of an answer: flush anything pending */
    if (R.incmd) rs_cmd_done();
    if (R.run_n) rs_flush_run(0);
    if (R.gdepth > 0) { R.gdepth = 0; ss_emit(R.supsub == 1, R.grp, R.gl); }
    while (CAPD > 0) { CAPD--; int n = CAPN[CAPD] < 0 ? 0 : CAPN[CAPD] > 1023 ? 1023 : CAPN[CAPD]; CAPB[CAPD][n] = 0; fputs(CAPB[CAPD], stdout); }
    if (COLOR) fputs(C_RESET, stdout);
    memset(&R, 0, sizeof R);
}
static void render_tok(int id) {
    if (id < 0 || id >= NTOK) return;
    if (!RENDER) { put_tok(id, 0); return; }
    for (uint32_t i = 0; i < TK[id].len; i++) rs_char(TK[id].s[i]);
}
static void render_text(const char *s) { rs_begin(0); while (*s) rs_char((unsigned char)*s++); rs_end(); }

/* ------------------------------------------------------------------ sampling */

static uint64_t RNG = 0x9E3779B97F4A7C15ULL;
static float rnd(void) { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (float)(RNG >> 40) / 16777216.0f; }

static float TEMP = 0.7f, TOPP = 0.8f; static int TOPKS = 20;

/* the distribution we actually sample from: top-k, temperature, top-p */
static int dist(const float *lg, int *id, float *p) {
    int best = 0; for (int i = 1; i < VOCAB; i++) if (lg[i] > lg[best]) best = i;
    if (TEMP <= 0) { id[0] = best; p[0] = 1; return 1; }
    int K = TOPKS > 0 && TOPKS < 256 ? TOPKS : 64, n = 0;
    for (int i = 0; i < VOCAB; i++) {
        float v = lg[i];
        if (n < K) { int j = n++; while (j > 0 && p[j - 1] < v) { p[j] = p[j - 1]; id[j] = id[j - 1]; j--; } p[j] = v; id[j] = i; }
        else if (v > p[K - 1]) { int j = K - 1; while (j > 0 && p[j - 1] < v) { p[j] = p[j - 1]; id[j] = id[j - 1]; j--; } p[j] = v; id[j] = i; }
    }
    float mx = p[0], z = 0;
    for (int j = 0; j < n; j++) { p[j] = expf((p[j] - mx) / TEMP); z += p[j]; }
    float c = 0; int keep = n;
    for (int j = 0; j < n; j++) { p[j] /= z; c += p[j]; if (c >= TOPP) { keep = j + 1; break; } }
    for (int j = 0; j < keep; j++) p[j] /= c;
    return keep;
}
/* presence penalty: tokens already written in this conversation get a small push down,
   so the model doesn't copy its earlier answers (Qwen's recommended fix for compressed models) */
static float PRES = 1.5f; static unsigned char *SEEN;
static void seen_clear(void) { if (SEEN) memset(SEEN, 0, (size_t)VOCAB); }
static void seen_add(int t) { if (!SEEN) SEEN = xcalloc((size_t)VOCAB, 1); if (t >= 0 && t < VOCAB) SEEN[t] = 1; }
static void penalize(float *lg) {
    if (PRES <= 0 || !SEEN) return;
    for (int i = 0; i < VOCAB; i++) if (SEEN[i]) lg[i] -= PRES;
}
static int sample_from(const float *lg) {
    int id[256]; float p[256]; int n = dist(lg, id, p);
    float r = rnd(), a = 0;
    for (int j = 0; j < n; j++) { a += p[j]; if (r <= a) return id[j]; }
    return id[n - 1];
}
/* speculative check: keep guess d with probability p(d) - output stays exactly the model's own distribution */
static int accept_guess(const float *lg, int d) {
    int id[256]; float p[256]; int n = dist(lg, id, p);
    for (int j = 0; j < n; j++) if (id[j] == d) return rnd() < p[j] || p[j] >= 1.0f;
    return 0;
}

/* ------------------------------------------------------------------ guessing memory
 * Every word the model reads or writes goes into a token memory saved in the
 * plate. To guess ahead, find the last time the current 3 (or 2) words
 * appeared and propose what came next. The model checks all guesses in one pass. */

#define HBITS 22
static int *CORP; static long NCORP, CORP0, CAPCORP = 1L << 22; static int NO_LEARN;
static int32_t *HT3, *HT2; static int LAST[3] = {-1, -1, -1};
static int DRAFTK = 0;
static long N_PASS, N_DRAFT, N_ACCEPT;

static uint32_t hsh(int a, int b, int c) {
    uint64_t x = (uint64_t)(uint32_t)a * 0x9E3779B97F4A7C15ULL ^ (uint64_t)(uint32_t)b * 0xC2B2AE3D27D4EB4FULL ^ (uint64_t)(uint32_t)c * 0x165667B19E3779F9ULL;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ULL; x ^= x >> 29;
    return (uint32_t)(x & ((1u << HBITS) - 1));
}
static void corp_rehash(void) {
    memset(HT3, 0, (1u << HBITS) * 4); memset(HT2, 0, (1u << HBITS) * 4);
    for (long i = 2; i < NCORP; i++) {
        if (i >= 3) HT3[hsh(CORP[i - 3], CORP[i - 2], CORP[i - 1])] = (int32_t)i;
        HT2[hsh(CORP[i - 2], CORP[i - 1], -7)] = (int32_t)i;
    }
}
static void corp_add(int t) {
    LAST[0] = LAST[1]; LAST[1] = LAST[2]; LAST[2] = t;
    if (!CORP) return;
    if (NCORP >= CAPCORP) {                        /* full: drop the oldest half, keep everything recent */
        long keep = CAPCORP / 2;
        memmove(CORP, CORP + (NCORP - keep), (size_t)keep * sizeof(int));
        NCORP = keep; CORP0 = -1; corp_rehash();
    }
    long i = NCORP;
    if (i >= 3) HT3[hsh(CORP[i - 3], CORP[i - 2], CORP[i - 1])] = (int32_t)i;   /* this trigram is followed by position i */
    if (i >= 2) HT2[hsh(CORP[i - 2], CORP[i - 1], -7)] = (int32_t)i;
    CORP[NCORP++] = t;
}
static int draft(int *d, int k) {
    if (!CORP || LAST[2] < 0) return 0;
    long p = -1;
    if (LAST[0] >= 0) { long q = HT3[hsh(LAST[0], LAST[1], LAST[2])]; if (q >= 3 && q < NCORP && CORP[q - 3] == LAST[0] && CORP[q - 2] == LAST[1] && CORP[q - 1] == LAST[2]) p = q; }
    if (p < 0 && LAST[1] >= 0) { long q = HT2[hsh(LAST[1], LAST[2], -7)]; if (q >= 2 && q < NCORP && CORP[q - 2] == LAST[1] && CORP[q - 1] == LAST[2]) p = q; }
    if (p < 0) return 0;
    int n = 0; while (n < k && p + n < NCORP) { d[n] = CORP[p + n]; n++; }
    return n;
}
static void corp_path(char *p, size_t n) { snprintf(p, n, "%s/draft.tok", PLATE); }
static void corp_load(void) {
    CORP = xcalloc((size_t)CAPCORP, sizeof(int)); HT3 = xcalloc(1u << HBITS, 4); HT2 = xcalloc(1u << HBITS, 4);
    if (NO_LEARN) return;                          /* in-session guessing only */
    char p[8192]; corp_path(p, sizeof p); FILE *f = fopen(p, "rb"); if (!f) return;
    uint32_t h[2]; long n = 0;
    if (fread(h, 4, 2, f) == 2 && !memcmp(h, "HMDR", 4)) {
        n = h[1] < CAPCORP ? (long)h[1] : CAPCORP;
        int *tmp = xcalloc((size_t)n + 1, sizeof(int));
        n = (long)fread(tmp, sizeof(int), (size_t)n, f);
        for (long i = 0; i < n; i++) corp_add(tmp[i]);
        free(tmp);
    }
    fclose(f); CORP0 = NCORP; LAST[0] = LAST[1] = LAST[2] = -1;
}
static void corp_save(void) {
    if (!CORP || NCORP == CORP0 || NO_LEARN) return;
    char p[8192], tp[8300]; corp_path(p, sizeof p); snprintf(tp, sizeof tp, "%s.part", p);
    FILE *f = fopen(tp, "wb"); if (!f) return;
    uint32_t h[2] = {0, (uint32_t)NCORP}; memcpy(h, "HMDR", 4);
    int ok = fwrite(h, 4, 2, f) == 2 && fwrite(CORP, sizeof(int), (size_t)NCORP, f) == (size_t)NCORP;
    ok = fflush(f) == 0 && fsync(fileno(f)) == 0 && ok; ok = fclose(f) == 0 && ok;
    if (ok && rename(tp, p) == 0) CORP0 = NCORP; else unlink(tp);
}
static void snaps_ensure(int k) {                  /* room to rewind the linear memories to any of k+1 positions */
    static int have = 0;
    if (k + 1 <= have) return;
    int kdim = LKH * LKD, cdim = 2 * kdim + LVH * LVD;
    if (!SNAPS) { SNAPS = xcalloc((size_t)(MAXK + 1) * NL, sizeof(float *)); SNAPC = xcalloc((size_t)(MAXK + 1) * NL, sizeof(float *)); }
    for (int t = have; t <= k; t++) for (int l = 0; l < NL; l++) if (!LY[l].full) {
        SNAPS[(size_t)t * NL + l] = xcalloc((size_t)LVH * LKD * LVD, 4);
        SNAPC[(size_t)t * NL + l] = xcalloc((size_t)cdim * CONVK, 4);
    }
    have = k + 1;
}
static void rewind_to(int t) {
    int cdim = 2 * LKH * LKD + LVH * LVD;
    for (int l = 0; l < NL; l++) if (!LY[l].full) {
        memcpy(SREC[l], SNAPS[(size_t)t * NL + l], (size_t)LVH * LKD * LVD * 4);
        memcpy(SCONV[l], SNAPC[(size_t)t * NL + l], (size_t)cdim * CONVK * 4);
    }
}

/* ------------------------------------------------------------------ settings the trainer tunes */

static void tuned_path(char *p, size_t n) { snprintf(p, n, "%s/tuned.cfg", PLATE); }
static void tuned_load(void) {
    char p[8192], line[256]; tuned_path(p, sizeof p); FILE *f = fopen(p, "r"); if (!f) return;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "draft_k=", 8)) DRAFTK = atoi(line + 8);
        if (!strncmp(line, "min_share=", 10)) MINW = (float)atof(line + 10);
    }
    fclose(f);
    if (DRAFTK < 0) DRAFTK = 0;
    if (DRAFTK > MAXK) DRAFTK = MAXK;
}
static void tuned_save(int k, float mw) {
    char p[8192], tp[8300]; tuned_path(p, sizeof p); snprintf(tp, sizeof tp, "%s.part", p);
    FILE *f = fopen(tp, "w"); if (!f) return;
    fprintf(f, "draft_k=%d\nmin_share=%.3f\n", k, mw);
    int ok = fflush(f) == 0 && fsync(fileno(f)) == 0; ok = fclose(f) == 0 && ok;
    if (ok) rename(tp, p); else unlink(tp);
}

/* ------------------------------------------------------------------ writing */

static int THINKING = 0, MAXGEN = 2048, PENDING_END = 0, ECHO = 1;
static double T_PRE, T_GEN; static long N_PRE, N_GEN; static double S_BEST, S_BESTW; static long S_ANS, S_WORDS, A_WORDS; static int W_IN;
static int IN_THINK;
static char LASTTEXT[2048]; static int LASTLEN;

/* answer length: a note the model sees with each question, plus a hard stop at a sentence end */
static int LEN_MODE = 2, LINE_CAP = -1, COLS = 80, V_LINES, V_COL;
static const char *LEN_NAME[] = {"full", "short", "medium", "long"};
static const char *LEN_NOTE[] = {"", "Answer in one short paragraph.", "Answer in at most three short paragraphs.", "Answer in detail, but stay focused."};
static const int LEN_CAP[] = {0, 8, 20, 60};
static int line_cap(void) { return LINE_CAP >= 0 ? LINE_CAP : LEN_CAP[LEN_MODE]; }
static void len_path(char *p, size_t n) { snprintf(p, n, "%s/chat.cfg", PLATE); }
static int SEARCH = 1;
static void len_save(void) {
    char p[8192]; len_path(p, sizeof p); FILE *f = fopen(p, "w"); if (!f) return;
    fprintf(f, "length=%s\nlines=%d\nsearch=%s\n", LEN_NAME[LEN_MODE], LINE_CAP, SEARCH ? "on" : "off"); fclose(f);
}
static void len_set(const char *name) { for (int i = 0; i < 4; i++) if (!strcmp(name, LEN_NAME[i])) { LEN_MODE = i; LINE_CAP = -1; } }
static void len_load(void) {
    char p[8192], line[128]; len_path(p, sizeof p); FILE *f = fopen(p, "r"); if (!f) return;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strncmp(line, "length=", 7)) { int c = LINE_CAP; len_set(line + 7); LINE_CAP = c; }
        if (!strncmp(line, "lines=", 6)) LINE_CAP = atoi(line + 6);
        if (!strncmp(line, "search=", 7)) SEARCH = !strcmp(line + 7, "on");
    }
    fclose(f);
}

static int emit(int t) {                           /* show one word; 1 = the answer is finished */
    if (t == IM_END || t == EOT) return 1;
    if (t == THINK_END) { if (IN_THINK && ECHO) { rs_end(); printf("\n"); rs_begin(0); } IN_THINK = 0; }
    else if (t != THINK) {
        if (ECHO) render_tok(t);
        if (ECHO && !IN_THINK && t >= 0 && t < NTOK)                     /* real word count of the visible answer */
            for (uint32_t i = 0; i < TK[t].len; i++) {
                unsigned char c = TK[t].s[i]; int sp = c == ' ' || c == '\n' || c == '\t' || c == '\r';
                if (!sp && !W_IN && (c & 0xC0) != 0x80) { A_WORDS++; S_WORDS++; }
                W_IN = !sp;
            }
        if (ECHO && !IN_THINK && t >= 0 && t < NTOK && line_cap() > 0) {     /* count screen lines of the answer */
            int end_sent = 0;
            for (uint32_t i = 0; i < TK[t].len; i++) {
                char c = (char)TK[t].s[i];
                if (c == '\n') { V_LINES++; V_COL = 0; end_sent = 1; }
                else { if ((TK[t].s[i] & 0xC0) != 0x80 && ++V_COL >= COLS) { V_LINES++; V_COL = 0; } if (c == '.' || c == '!' || c == '?') end_sent = 1; }
            }
            if ((V_LINES >= line_cap() && end_sent) || V_LINES >= line_cap() + 3) { fflush(stdout); N_GEN++; return 1; }
        }
        if (t >= 0 && t < NTOK) for (uint32_t i = 0; i < TK[t].len && LASTLEN < (int)sizeof LASTTEXT - 1; i++) LASTTEXT[LASTLEN++] = (char)TK[t].s[i];
        LASTTEXT[LASTLEN] = 0;
    }
    if (ECHO) fflush(stdout);
    N_GEN++;
    return 0;
}

static void (*LIVE_HOOK)(int tokens, double secs);   /* trainer: live progress while a round is being written */
static void generate(void) {
    penalize(LOG);
    int t = sample_from(LOG), g = 0; double t0 = now(), tl = t0;
    LASTLEN = 0; LASTTEXT[0] = 0; V_LINES = 0; V_COL = 0; A_WORDS = 0; W_IN = 0;
    while (g < MAXGEN) {
        if (LIVE_HOOK && now() - tl > 2.0) { tl = now(); LIVE_HOOK(g, tl - t0); }
        if (emit(t)) break;
        g++; corp_add(t); seen_add(t); PENDING_END = 1;
        if (POS + MAXK + 6 >= CTX) break;
        int d[MAXK], k = DRAFTK > 0 ? draft(d, DRAFTK) : 0;
        if (k > MAXGEN - g) k = MAXGEN - g;
        if (k <= 0) { forward(t, 1); N_PASS++; penalize(LOG); t = sample_from(LOG); continue; }
        int in[MAXK + 1]; in[0] = t; memcpy(in + 1, d, (size_t)k * sizeof(int));
        snaps_ensure(k);
        long p0 = POS; SNAPON = 1; forward_batch(in, k + 1, 2); SNAPON = 0;
        N_PASS++; N_DRAFT += k;
        int j = 0, stop = 0;
        while (j < k && (penalize(LOGB + (size_t)j * VOCAB), accept_guess(LOGB + (size_t)j * VOCAB, d[j]))) {
            N_ACCEPT++;
            if (emit(d[j])) { stop = 1; break; }
            g++; corp_add(d[j]); seen_add(d[j]); j++;
            if (g >= MAXGEN) break;
        }
        rewind_to(j); POS = p0 + j + 1;
        if (stop || g >= MAXGEN) break;
        memcpy(LOG, LOGB + (size_t)j * VOCAB, (size_t)VOCAB * 4);
        if (j < k) LOG[d[j]] = -1e30f;                 /* rejected guess: draw from what's left (row already penalized) */
        else penalize(LOG);
        t = sample_from(LOG);
    }
}

static int build_turn(const char *msg, int *ids, int max) {
    int n = 0; char buf[64];
    if (PENDING_END) { ids[n++] = IM_END; n += encode("\n", ids + n, max - n); }   /* close the previous answer */
    ids[n++] = IM_START;
    size_t L = strlen(msg) + 96; char *u = xcalloc(L, 1);
    snprintf(u, L, "user\n%s%s%s", msg, LEN_NOTE[LEN_MODE][0] ? "\n\n" : "", LEN_NOTE[LEN_MODE]);
    n += encode(u, ids + n, max - n - 16); free(u);
    ids[n++] = IM_END; n += encode("\n", ids + n, max - n);
    ids[n++] = IM_START; snprintf(buf, sizeof buf, "assistant\n"); n += encode(buf, ids + n, max - n);
    if (THINK >= 0) {
        ids[n++] = THINK;
        if (THINKING) n += encode("\n", ids + n, max - n);
        else { n += encode("\n\n", ids + n, max - n); ids[n++] = THINK_END; n += encode("\n\n", ids + n, max - n); }
    }
    return n;
}


/* ------------------------------------------------------------------ local search over your fed files
 * Everything fed to the trainer is kept as text in plate/train/feeds/. When a question clearly
 * matches a passage there, the best one or two short passages ride along with the question,
 * so answers about your own material come from your own material. Nothing leaves the machine. */

typedef struct { int file; size_t off, len; int words; } Chunk;
static char *CORPUS_TXT; static size_t CORPUS_N; static Chunk *CH; static int NCH_, NFILES; static char FNAMES[256][96];
static const char *STOP[] = {"the","and","for","are","but","not","you","all","any","can","had","her","was","one","our","out","has","have",
    "him","his","how","its","may","new","now","old","see","two","way","who","did","get","let","say","she","too","use","what","when",
    "where","which","while","with","would","could","should","about","after","again","also","been","being","does","doing","each","from",
    "into","just","like","make","many","more","most","much","must","only","other","over","same","some","such","than","that","their",
    "them","then","there","these","they","this","those","through","very","were","will","your","yours","explain","tell","please","give",
    "why","is","in","of","to","a","an","it","on","me","my","be","do","so","if","or","as","at","by","we","i","know","think","want","need"};
static int is_stop(const char *w) { for (size_t i = 0; i < sizeof STOP / sizeof *STOP; i++) if (!strcmp(w, STOP[i])) return 1; return 0; }

static void search_load(void) {
    char dir[8400]; snprintf(dir, sizeof dir, "%s/train/feeds", PLATE);
    DIR *dp = opendir(dir); if (!dp) return;
    struct dirent *e; size_t cap = 0;
    while ((e = readdir(dp)) && NFILES < 256) {
        size_t L = strlen(e->d_name); if (L < 5 || strcmp(e->d_name + L - 4, ".txt")) continue;
        char p[9000]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        FILE *f = fopen(p, "rb"); if (!f) continue;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        if (n <= 0 || n > (32L << 20)) { fclose(f); continue; }
        if (CORPUS_N + (size_t)n + 2 > cap) { cap = (CORPUS_N + (size_t)n + 2) * 2; CORPUS_TXT = realloc(CORPUS_TXT, cap); if (!CORPUS_TXT) die("out of memory"); }
        size_t got = fread(CORPUS_TXT + CORPUS_N, 1, (size_t)n, f); fclose(f);
        int fi = NFILES; snprintf(FNAMES[NFILES++], 96, "%.*s", (int)(L - 4 > 90 ? 90 : L - 4), e->d_name);
        size_t a = CORPUS_N, end = CORPUS_N + got; CORPUS_N = end; CORPUS_TXT[CORPUS_N++] = '\n';
        while (a < end) {                                  /* ~70-word passages */
            size_t b = a; int w = 0, in = 0;
            while (b < end && w < 70) { int sp = isspace((unsigned char)CORPUS_TXT[b]); if (!sp && !in) w++; in = !sp; b++; }
            while (b < end && !isspace((unsigned char)CORPUS_TXT[b])) b++;
            static int capc = 0;
            if (NCH_ >= capc) { capc = capc ? capc * 2 : 4096; CH = realloc(CH, (size_t)capc * sizeof(Chunk)); if (!CH) die("out of memory"); }
            CH[NCH_++] = (Chunk){fi, a, b - a, w};
            a = b;
        }
    }
    closedir(dp);
}

/* find passages; returns how many were added to out (as "[file] text" lines) */
static int search_find(const char *q, char *out, size_t outn, char *names, size_t nn) {
    out[0] = 0; names[0] = 0;
    if (!SEARCH || !NCH_) return 0;
    char kw[12][32]; int nk = 0;
    for (const char *s = q; *s && nk < 12;) {               /* keywords from the question */
        while (*s && !isalnum((unsigned char)*s)) s++;
        char w[32]; int L = 0;
        while (*s && isalnum((unsigned char)*s)) { if (L < 31) w[L++] = (char)tolower((unsigned char)*s); s++; }
        w[L] = 0;
        if (L >= 3 && !is_stop(w)) { int dup = 0; for (int i = 0; i < nk; i++) if (!strcmp(kw[i], w)) dup = 1; if (!dup) strcpy(kw[nk++], w); }
    }
    if (!nk) return 0;
    unsigned char *tf = xcalloc((size_t)NCH_ * nk, 1); int df[12] = {0}; double avg = 0;
    for (int c = 0; c < NCH_; c++) {
        avg += CH[c].words; const char *s = CORPUS_TXT + CH[c].off, *e = s + CH[c].len;
        while (s < e) {
            while (s < e && !isalnum((unsigned char)*s)) s++;
            char w[32]; int L = 0;
            while (s < e && isalnum((unsigned char)*s)) { if (L < 31) w[L++] = (char)tolower((unsigned char)*s); s++; }
            w[L] = 0; if (L < 3) continue;
            for (int k = 0; k < nk; k++) if (!strcmp(w, kw[k])) { if (!tf[(size_t)c * nk + k]) df[k]++; if (tf[(size_t)c * nk + k] < 255) tf[(size_t)c * nk + k]++; }
        }
    }
    avg /= NCH_;
    int best[2] = {-1, -1}; double bs[2] = {0, 0};
    for (int c = 0; c < NCH_; c++) {
        double sc = 0; int hit = 0;
        for (int k = 0; k < nk; k++) {
            int t = tf[(size_t)c * nk + k]; if (!t) continue; hit++;
            double idf = log(1.0 + (NCH_ - df[k] + 0.5) / (df[k] + 0.5));
            sc += idf * t * 2.2 / (t + 1.2 * (0.25 + 0.75 * CH[c].words / avg));
        }
        int need = nk >= 2 ? 2 : 1;                          /* a real match, not one common word */
        if (hit < need) continue;
        if (nk == 1 && df[0] * 50 > NCH_) continue;
        if (sc > bs[0]) { bs[1] = bs[0]; best[1] = best[0]; bs[0] = sc; best[0] = c; } else if (sc > bs[1]) { bs[1] = sc; best[1] = c; }
    }
    free(tf);
    int n = 0; size_t o = 0, on = 0;
    for (int i = 0; i < 2; i++) {
        int c = best[i]; if (c < 0 || (i == 1 && bs[1] < bs[0] * 0.6)) continue;
        o += (size_t)snprintf(out + o, outn - o, "[%s] %.*s\n", FNAMES[CH[c].file], (int)(CH[c].len > 600 ? 600 : CH[c].len), CORPUS_TXT + CH[c].off);
        if (!strstr(names, FNAMES[CH[c].file])) on += (size_t)snprintf(names + on, nn - on, "%s%s", on ? ", " : "", FNAMES[CH[c].file]);
        n++;
        if (o >= outn - 1) break;
    }
    for (char *p = out; *p; p++) if (*p == '\n' && p[1] && p[1] != '[') *p = ' ';   /* one line per passage */
    return n;
}

static long LA_TOK, LA_WORDS; static double LA_SEC;
static void answer(const char *msg) {
    USE_TRIM = utf8_ok_run((const uint8_t *)msg, (uint32_t)strlen(msg));   /* other scripts: score the full vocabulary */
    int max = (int)strlen(msg) * 2 + 160, *ids = xcalloc((size_t)max, sizeof(int));
    int n = build_turn(msg, ids, max);
    if (POS + n + 16 > CTX) { if (ECHO) printf("[conversation is full - starting fresh]\n"); state_reset(); seen_clear(); PENDING_END = 0; free(ids); answer(msg); return; }
    double t0 = now();
    forward_batch(ids, n, 1);
    for (int i = 0; i < n; i++) corp_add(ids[i]);
    T_PRE += now() - t0; N_PRE += n; free(ids);
    IN_THINK = THINKING; if (ECHO) rs_begin(IN_THINK);
    long g0 = N_GEN;
    t0 = now();
    generate();
    if (ECHO) rs_end();
    double dt = now() - t0; T_GEN += dt;
    if (ECHO) {
        S_ANS++; LA_TOK = N_GEN - g0; LA_WORDS = A_WORDS; LA_SEC = dt;
        if (LA_TOK >= 50 && dt > 0) { if (LA_TOK / dt > S_BEST) S_BEST = LA_TOK / dt; if (A_WORDS / dt > S_BESTW) S_BESTW = A_WORDS / dt; }
        printf("\n%s[%ld tokens, %ld words in %.1fs - %.2f tok/s, %.2f words/s]%s\n", COLOR ? "\033[2m" : "", LA_TOK, LA_WORDS, dt,
               dt > 0 ? LA_TOK / dt : 0, dt > 0 ? LA_WORDS / dt : 0, COLOR ? C_RESET : "");
    }
}


/* what the chat window calls: local model, with your files' best passages when relevant */
static void chat_answer(const char *msg) {
    static char passages[1600], names[400];
    int np = search_find(msg, passages, sizeof passages, names, sizeof names);
    if (np) {
        if (ECHO) printf("%s[from your files: %s]%s\n", COLOR ? "\033[2m" : "", names, COLOR ? C_RESET : "");
        size_t L = strlen(passages) + strlen(msg) + 128; char *aug = xcalloc(L, 1);
        snprintf(aug, L, "Notes from my files (use them only if relevant):\n%s\nQuestion: %s", passages, msg);
        answer(aug); free(aug);
    } else answer(msg);
}

/* all-time totals, kept in plate/lifetime.cfg */
static double LT_GEN, LT_GSEC, LT_PRE, LT_PSEC, LT_BEST, LT_WORDS, LT_BESTW; static long LT_ANS; static int LT_LOADED;
static void lt_path(char *p, size_t n) { snprintf(p, n, "%s/lifetime.cfg", PLATE); }
static void lt_load(void) {
    char p[8192], line[128]; lt_path(p, sizeof p); LT_LOADED = 1; FILE *f = fopen(p, "r"); if (!f) return;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "gen_tokens=", 11)) LT_GEN = atof(line + 11);
        if (!strncmp(line, "gen_secs=", 9)) LT_GSEC = atof(line + 9);
        if (!strncmp(line, "read_tokens=", 12)) LT_PRE = atof(line + 12);
        if (!strncmp(line, "read_secs=", 10)) LT_PSEC = atof(line + 10);
        if (!strncmp(line, "best_tps=", 9)) LT_BEST = atof(line + 9);
        if (!strncmp(line, "answers=", 8)) LT_ANS = atol(line + 8);
        if (!strncmp(line, "words=", 6)) LT_WORDS = atof(line + 6);
        if (!strncmp(line, "best_wps=", 9)) LT_BESTW = atof(line + 9);
    }
    fclose(f);
}
static void lt_save(void) {
    if (!LT_LOADED || (!N_GEN && !N_PRE)) return;
    char p[8192], tp[8300]; lt_path(p, sizeof p); snprintf(tp, sizeof tp, "%s.part", p);
    FILE *f = fopen(tp, "w"); if (!f) return;
    fprintf(f, "gen_tokens=%.0f\ngen_secs=%.2f\nread_tokens=%.0f\nread_secs=%.2f\nbest_tps=%.3f\nanswers=%ld\nwords=%.0f\nbest_wps=%.3f\n",
            LT_GEN + N_GEN, LT_GSEC + T_GEN, LT_PRE + N_PRE, LT_PSEC + T_PRE, S_BEST > LT_BEST ? S_BEST : LT_BEST, LT_ANS + S_ANS,
            LT_WORDS + S_WORDS, S_BESTW > LT_BESTW ? S_BESTW : LT_BESTW);
    fclose(f); rename(tp, p);
    LT_GEN += N_GEN; LT_GSEC += T_GEN; LT_PRE += N_PRE; LT_PSEC += T_PRE; LT_ANS += S_ANS; LT_WORDS += S_WORDS;
    if (S_BEST > LT_BEST) LT_BEST = S_BEST;
    if (S_BESTW > LT_BESTW) LT_BESTW = S_BESTW;
    N_GEN = 0; T_GEN = 0; N_PRE = 0; T_PRE = 0; S_ANS = 0; S_WORDS = 0;
}
static void save_all(void) { heat_save(); corp_save(); lt_save(); }

static void stats(void) {
    long used = 0;
    for (long i = 0; i < (long)NL * NE; i++) if (HEAT[i] || HEAT_ALL[i]) used++;
    double all = T_DENSE + T_EXP + T_ATT + T_HEAD; if (all <= 0) all = 1;
    double ag = LT_GEN + N_GEN, as = LT_GSEC + T_GEN, best = S_BEST > LT_BEST ? S_BEST : LT_BEST;
    printf("  this session: writing %ld tokens (%ld words) in %.1fs, %.2f tok/s (%.2f words/s), reading %ld tokens at %.2f tok/s\n",
           N_GEN, S_WORDS, T_GEN, T_GEN > 0 ? N_GEN / T_GEN : 0, T_GEN > 0 ? S_WORDS / T_GEN : 0, N_PRE, T_PRE > 0 ? N_PRE / T_PRE : 0);
    double aw = LT_WORDS + S_WORDS, bestw = S_BESTW > LT_BESTW ? S_BESTW : LT_BESTW;
    printf("  all time    : %.0f tokens (%.0f words) over %ld answers, average %.2f tok/s (%.2f words/s)\n",
           ag, aw, LT_ANS + S_ANS, as > 0 ? ag / as : 0, as > 0 ? aw / as : 0);
    printf("  best ever   : %.2f tok/s, %.2f words/s (answers of 50+ tokens)\n", best, bestw);
    printf("  guessing: %d ahead, %ld of %ld guesses landed (%.0f%%), %.2f tokens per pass\n", DRAFTK, N_ACCEPT, N_DRAFT,
           N_DRAFT ? 100.0 * N_ACCEPT / N_DRAFT : 0, N_PASS ? (double)N_GEN / N_PASS : 0);
    printf("  time    : attention weights %.0f%%, experts %.0f%%, memory math %.0f%%, word choice %.0f%%\n",
           100 * T_DENSE / all, 100 * T_EXP / all, 100 * T_ATT / all, 100 * T_HEAD / all);
    if (NE) printf("  RAM hits: %.0f%% of expert calls found their expert already in RAM; next-layer prediction right %.0f%% of the time\n", N_PROBE ? 100.0 * N_HIT / N_PROBE : 0, N_PRED ? 100.0 * N_PRED_HIT / N_PRED : 0);
    printf("  memory  : %ld of %d positions used (linear layers never grow), guessing memory %ld words\n", POS, CTX, NCORP);
    if (NE) printf("  experts : %ld of %d ever woken, %ld warmed at boot (%.1f GB), skip below %.2f share\n", used, NL * NE, PRELOADED, PRELOAD_BYTES / 1e9, MINW);
    printf("  mode    : %s experts, %s attention, thinking %s, temp %.2f\n", FLOAT_ACT ? "bit-walk" : "lookup-table", Q8 ? "int8" : "bf16", THINKING ? "on" : "off", TEMP);
}

static void set_think(int on) {
    if (on && THINK < 0 && NL) { printf("[this model has no thinking mode]\n"); on = 0; }
    THINKING = on;
    if (on) { TEMP = 0.6f; TOPP = 0.95f; } else { TEMP = 0.7f; TOPP = 0.8f; }
}


/* ------------------------------------------------------------------ /card: one screen, every number measured */

static double BOOT_SECS;
static void hw_info(char *model, size_t mn, char *cpu, size_t cn, int *cores, double *ram) {
    snprintf(model, mn, "this computer"); snprintf(cpu, cn, "CPU"); *cores = (int)sysconf(_SC_NPROCESSORS_ONLN); *ram = phys_ram();
#if defined(__APPLE__)
    size_t l = mn; sysctlbyname("hw.model", model, &l, NULL, 0);
    char b[256] = {0}; l = sizeof b - 1;
    if (sysctlbyname("machdep.cpu.brand_string", b, &l, NULL, 0) == 0) snprintf(cpu, cn, "%s", b);
    int pc = 0; l = sizeof pc; if (sysctlbyname("hw.physicalcpu", &pc, &l, NULL, 0) == 0 && pc > 0) *cores = pc;
#else
    FILE *f = fopen("/proc/cpuinfo", "r"); char line[512];
    if (f) { while (fgets(line, sizeof line, f)) if (!strncmp(line, "model name", 10)) { char *c = strchr(line, ':'); if (c) { snprintf(cpu, cn, "%s", c + 2); cpu[strcspn(cpu, "\n")] = 0; } break; } fclose(f); }
    snprintf(model, mn, "Linux machine");
#endif
    char *q;                                                         /* tidy the CPU name */
    while ((q = strstr(cpu, "(R)")) || (q = strstr(cpu, "(TM)"))) memmove(q, q + (q[1] == 'R' ? 3 : 4), strlen(q + (q[1] == 'R' ? 3 : 4)) + 1);
    if ((q = strstr(cpu, " CPU @"))) { memmove(q, q + 4, strlen(q + 4) + 1); }
    while ((q = strstr(cpu, "  "))) memmove(q, q + 1, strlen(q + 1) + 1);
}
static double dir_bytes(const char *sub) {
    char d[8400]; snprintf(d, sizeof d, "%s/%s", PLATE, sub); DIR *dp = opendir(d); if (!dp) return 0;
    double t = 0; struct dirent *e; char p[9000]; struct stat sb;
    while ((e = readdir(dp))) { if (e->d_name[0] == '.') continue; snprintf(p, sizeof p, "%s/%s", d, e->d_name); if (stat(p, &sb) == 0 && S_ISREG(sb.st_mode)) t += (double)sb.st_size; }
    closedir(dp); return t;
}
static int term_cols(void) { struct winsize w; if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_col > 20) return w.ws_col; return 100; }
static void rule(const char *title, int w) {
    int n = printf("\033[1m== %s ", title); n -= 4;
    for (int i = n; i < w; i++) putchar('=');
    printf("\033[0m\n");
}
static const char *num(double v, char *b) {                          /* 12345 -> 12,345 */
    char t[64]; snprintf(t, sizeof t, "%.0f", v); int L = (int)strlen(t), k = 0;
    for (int i = 0; i < L; i++) { b[k++] = t[i]; if ((L - i - 1) % 3 == 0 && i < L - 1) b[k++] = ','; } b[k] = 0; return b;
}
static void card(void) {
    char model[128], cpu[256], b1[32], b2[32], b3[32]; int cores; double ram;
    hw_info(model, sizeof model, cpu, sizeof cpu, &cores, &ram);
    int w = term_cols(); if (w > 120) w = 120;
    double exp_params = (double)NL * NE * 3.0 * MOEI * DIM, emb = (double)EMB.shape[0] * EMB.shape[1];
    double total = DENSE_PARAMS + exp_params, active = DENSE_PARAMS - emb + (NE ? exp_params * TOPK / NE : 0);
    double full_gb = total * 2 / 1e9, plate_gb = (dir_bytes("experts") + dir_bytes("dense")) / 1e9, q8_gb = dir_bytes("q8") / 1e9;
    double ag = LT_GEN + N_GEN, as = LT_GSEC + T_GEN, aw = LT_WORDS + S_WORDS;
    double best = S_BEST > LT_BEST ? S_BEST : LT_BEST, bestw = S_BESTW > LT_BESTW ? S_BESTW : LT_BESTW;
    double all = T_DENSE + T_EXP + T_ATT + T_HEAD; if (all <= 0) all = 1;
    long used = 0; for (long i = 0; i < (long)NL * NE; i++) if (HEAT[i] || HEAT_ALL[i]) used++;
    char title[512]; snprintf(title, sizeof title, "HoloMoE | %.1fB-parameter %s | %s | %s | %d cores | %.0f GB RAM", total / 1e9, NE ? "mixture-of-experts" : "dense model", model, cpu, cores, ram / 1073741824.0);
    rule(title, w);
    if (NE) printf(" model    : %.1fB parameters, %.1fB used per token (%d of %d experts per layer, %d layers; text model; %s)\n", total / 1e9, active / 1e9, TOPK, NE, NL, ARCH);
    else printf(" model    : %.1fB parameters, all used every token (dense, %d layers; text model; %s)\n", total / 1e9, NL, ARCH);
    printf(" size     : %.0f GB at full precision | plate on SSD %.1f GB | fast-boot copy %.1f GB | %.1fx bigger than this computer's RAM\n", full_gb, plate_gb, q8_gb, full_gb / (ram / 1e9));
    printf(" RAM      : %.2f GB attention held | %.2f GB of hot experts cached (the OS can reclaim it) | booted in %.1fs\n", Q8_BYTES / 1e9, PRELOAD_BYTES / 1e9, BOOT_SECS);
    if (LA_TOK) printf(" answer   : %ld tokens, %ld words in %.1fs = %.2f tok/s, %.2f words/s\n", LA_TOK, LA_WORDS, LA_SEC, LA_SEC > 0 ? LA_TOK / LA_SEC : 0, LA_SEC > 0 ? LA_WORDS / LA_SEC : 0);
    else printf(" answer   : (ask a question first)\n");
    printf(" session  : %s tokens, %s words | writing %.2f tok/s, %.2f words/s | reading %.2f tok/s\n", num(N_GEN, b1), num(S_WORDS, b2),
           T_GEN > 0 ? N_GEN / T_GEN : 0, T_GEN > 0 ? S_WORDS / T_GEN : 0, T_PRE > 0 ? N_PRE / T_PRE : 0);
    printf(" all time : %s tokens, %s words over %ld answers | average %.2f tok/s, %.2f words/s\n", num(ag, b1), num(aw, b2), LT_ANS + S_ANS,
           as > 0 ? ag / as : 0, as > 0 ? aw / as : 0);
    printf("\033[1m BEST     : %.2f tok/s | %.2f words/s | %.0f words a minute\033[0m\n", best, bestw, bestw * 60);
    printf(" guessing : %d ahead | %ld of %ld landed (%.0f%%) | %.2f tokens per pass | guessing memory %s words\n", DRAFTK, N_ACCEPT, N_DRAFT,
           N_DRAFT ? 100.0 * N_ACCEPT / N_DRAFT : 0, N_PASS ? (double)N_GEN / N_PASS : 0, num(NCORP, b3));
    if (NE) printf(" experts  : %ld of %d used so far | %ld warm in RAM | found in RAM %.0f%% | next-layer prediction right %.0f%%\n", used, NL * NE, PRELOADED,
           N_PROBE ? 100.0 * N_HIT / N_PROBE : 0, N_PRED ? 100.0 * N_PRED_HIT / N_PRED : 0);
    printf(" time     : attention weights %.0f%% | experts %.0f%% | memory math %.0f%% | word choice %.0f%%\n",
           100 * T_DENSE / all, 100 * T_EXP / all, 100 * T_ATT / all, 100 * T_HEAD / all);
    printf(" engine   : %s experts | %s attention | vocab %s | %s | prefetch %s | thinking %s | temp %.2f | repeat penalty %.1f\n",
           FLOAT_ACT ? "bit-walk" : "lookup-table ternary", !Q8 ? "bf16" : LEAN ? "int4" : "int8", TRIM ? "trimmed" : "full", SIMD ? "AVX2" : "scalar", PREDICT ? "on" : "off", THINKING ? "on" : "off", TEMP, PRES);
    printf(" extras   : fully offline | file search %s (%d passages, %d files) | vocab %s%s\n", 
           SEARCH && NCH_ ? "on" : "off", NCH_, NFILES, TRIM ? "trimmed" : "full", TRIM ? "" : "");
    printf(" for scale: published Raspberry Pi 5 record (ByteShape): 8.03 tok/s, 30B model held fully in 16 GB RAM\n");
    for (int i = 0; i < w; i++) putchar('=');
    putchar('\n');
}

static void help(void) {
    puts("CHAT COMMANDS\n"
         "  /think on | off            show its reasoning (better for math, logic, code)\n"
         "  /short /medium /long /full answer length (saved)     /lines N  exact line limit\n"
         "  /stats                     this session + all-time speed, guessing, RAM hits\n"
         "  /card                      one-screen summary of the machine, model and every speed\n"
         "  /search on | off           use passages from the files you fed it (on by default)\n"
         "  /reset   fresh conversation      /save   save learning      /quit   exit + save\n"
         "START OPTIONS  (./holomoe [option ...])\n"
         "  -p \"question\"   one answer, then exit       --think       start with reasoning on\n"
         "  --model NAME    35b (default), 2b, 30b*    (*untested; get one: python3 shardpress.py models)\n"
         "  --temp 0.7      creativity: 0 = exact, 0.3 focused, 0.7 normal, 1.0+ wild\n"
         "  --seed N        repeatable answers           -n 200        max tokens per answer\n"
         "  --repeat-penalty 1.5   discourages repeating earlier answers (0 = off)\n"
         "  --plain         no colors        --raw   show the model's raw Markdown/LaTeX\n"
         "  --lean          4-bit attention: about half the RAM (first run builds plate/q4/)\n"
         "  --full-vocab    score all 248k tokens (normally trimmed to Latin/math/code, auto-off for other scripts)\n"
         "  --length short|medium|long|full              --lines 12    line limit\n"
         "  --ctx 4096      conversation memory (tokens) --preload GB  experts warmed at boot\n"
         "  --guess N  --skip 0.04   speed settings (the trainer tunes these for you)\n"
         "  --no-learn      don't load or save the guessing memory this session\n"
         "TESTING\n"
         "  --tokenize \"text\"   token numbers      --ppl file.txt   quality score (lower = better)\n"
         "  --bf16  --float-act  --no-simd  --no-predict   slower exact/reference modes for checking\n"
         "LEARNS AUTOMATICALLY  (in plate/)\n"
         "  heat.bin   which experts to keep warm      draft.tok   guessing memory from chats + files\n"
         "  tuned.cfg  fastest settings (trainer)      chat.cfg    your answer length   q8/  fast boot\n"
         "TRAINING  (python3 holotrain.py --help for all)\n"
         "  python3 holotrain.py 10m --turbo           train; window opens with live stats + record\n"
         "  python3 holotrain.py --feed paper.pdf      feed any file: PDF, Word, Markdown, code, images");
}

static void repl(void) {
    char shape[96]; if (NE) snprintf(shape, sizeof shape, "%d experts each (top %d)", NE, TOPK); else snprintf(shape, sizeof shape, "dense");
    printf("%sHoloMoE%s %s- %s, %d layers, %s, vocab %d, guessing %d ahead%s\n", COLOR ? "\033[1;35m" : "", COLOR ? C_RESET : "",
           COLOR ? "\033[2m" : "", ARCH, NL, shape, VOCAB, DRAFTK, COLOR ? C_RESET : "");
    printf("type /help for all commands%s\n", NCH_ ? "  |  file search on" : "");
    if (line_cap() > 0) printf("answers: %s, up to about %d lines\n", LEN_NAME[LEN_MODE], line_cap()); else printf("answers: full length\n");
    char line[65536];
    for (;;) {
        printf("\n%syou>%s ", COLOR ? "\033[1;32m" : "", COLOR ? C_RESET : ""); fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) break;
        if (!strcmp(line, "/save")) { save_all(); printf("[heat map + guessing memory saved]\n"); continue; }
        if (!strcmp(line, "/reset")) { state_reset(); seen_clear(); PENDING_END = 0; LAST[0] = LAST[1] = LAST[2] = -1; printf("[fresh conversation]\n"); continue; }
        if (!strcmp(line, "/stats")) { stats(); continue; }
        if (!strcmp(line, "/help") || !strcmp(line, "/?")) { help(); continue; }
        if (!strcmp(line, "/card")) { card(); continue; }
        if (!strncmp(line, "/search", 7)) {
            if (strstr(line, "off")) SEARCH = 0; else if (strstr(line, "on")) SEARCH = 1; else SEARCH = !SEARCH;
            len_save(); printf("[file search %s - %d passages from %d files]\n", SEARCH ? "on" : "off", NCH_, NFILES); continue;
        }
        if (!strcmp(line, "/short") || !strcmp(line, "/medium") || !strcmp(line, "/long") || !strcmp(line, "/full")) {
            len_set(line + 1); len_save();
            if (line_cap() > 0) printf("[answers: %s, up to about %d lines]\n", LEN_NAME[LEN_MODE], line_cap()); else printf("[answers: full length]\n");
            continue;
        }
        if (!strncmp(line, "/lines", 6)) { LINE_CAP = atoi(line + 6); if (LINE_CAP < 0) LINE_CAP = 0; len_save(); printf("[answers stop at about %d lines%s]\n", LINE_CAP, LINE_CAP ? "" : " - no limit"); continue; }
        if (!strncmp(line, "/think", 6)) { set_think(strstr(line, "on") != NULL); printf("[thinking %s]\n", THINKING ? "on" : "off"); continue; }
        if (line[0] == '/' && isalpha((unsigned char)line[1])) { printf("[unknown command %s - type /help for the list]\n", strtok(line, " ")); continue; }
        printf("%sholo>%s ", COLOR ? "\033[1;36m" : "", COLOR ? C_RESET : ""); fflush(stdout);
        chat_answer(line);
    }
}

/* ------------------------------------------------------------------ quality check: perplexity */

static double perplexity(const char *text) {      /* lower = the model predicts real text better (full vocabulary) */
    int keep_trim = USE_TRIM; USE_TRIM = 0;
    int max = (int)strlen(text) + 8, *ids = xcalloc((size_t)max, sizeof(int)), n = encode(text, ids, max);
    state_reset();
    double nll = 0; int cnt = 0;
    for (int s = 0; s < n - 1; s += PB) {
        int B = n - 1 - s < PB ? n - 1 - s : PB;
        forward_batch(ids + s, B, 2);
        for (int t = 0; t < B; t++) {
            const float *lg = LOGB + (size_t)t * VOCAB; float mx = lg[0]; double z = 0;
            for (int i = 1; i < VOCAB; i++) if (lg[i] > mx) mx = lg[i];
            for (int i = 0; i < VOCAB; i++) z += exp((double)lg[i] - mx);
            nll += -(lg[ids[s + t + 1]] - mx - log(z)); cnt++;
        }
    }
    state_reset(); free(ids); USE_TRIM = keep_trim;
    return cnt ? exp(nll / cnt) : 0;
}

/* ------------------------------------------------------------------ trainer */

static const char *HOLDOUT =
    "The river carried silt down from the mountains every spring, and over thousands of years the silt built a wide, flat plain. "
    "Farmers learned to plant after the floods went down, because the new soil was dark and rich. To measure the land again after "
    "each flood, they invented simple tools: a rope with evenly spaced knots, a plumb line, and a set of rules for right angles. "
    "Those rules became geometry. In the same way, the need to predict the floods led people to watch the stars and keep careful "
    "calendars. Science often starts like this, with a practical problem that will not go away.";

static const char *TOPICS[] = {
    "how a refrigerator works", "the ocean at night", "staying focused while working", "weather versus climate",
    "how laws are made", "reversing a string in Python", "photosynthesis", "making vegetable soup", "why the sky is blue",
    "the Pythagorean theorem", "planting a tree", "the seasons on Earth", "black holes", "saving energy at home",
    "how vaccines work", "viruses and bacteria", "inviting a neighbor to dinner", "how computers store numbers",
    "brewing good coffee", "supply and demand", "how airplanes fly", "holograms", "learning guitar", "electric circuits",
    "fixing a squeaky hinge", "the water cycle", "compound interest", "lighthouses", "machine learning", "how bees make honey",
    "buying a used car", "volcanoes", "the human heart", "solar panels", "the printing press", "sourdough bread",
    "chess openings", "rainbows", "the Moon landing", "recycling plastic", "sleep and memory", "earthquakes",
    "the stock market", "growing tomatoes", "magnets", "the Roman Empire", "writing a resume", "batteries",
};
static const char *FORMS[] = {
    "Explain %s in simple terms.", "Give me three interesting facts about %s.", "Write a short paragraph about %s.",
    "What are common misconceptions about %s?", "How would you teach %s to a beginner?", "Write a short story involving %s.",
};
#define NTOPICS ((int)(sizeof TOPICS / sizeof *TOPICS))
#define NFORMS ((int)(sizeof FORMS / sizeof *FORMS))
static void practice_prompt(int r, char *out, size_t n) {   /* 288 different questions before any repeat */
    snprintf(out, n, FORMS[(r / NTOPICS) % NFORMS], TOPICS[(r * 7) % NTOPICS]);
}

static double BW_GBS, OPS_G;                       /* measured on this machine */
typedef struct { const uint64_t *buf; size_t n; uint64_t sums[NCH]; } BWJ;
static void bw_chunk(void *c, size_t i) {
    BWJ *j = c; size_t per = j->n / NCH, a = i * per; uint64_t s = 0;
    for (size_t k = a; k < a + per; k++) s += j->buf[k];
    j->sums[i] = s;
}
typedef struct { const int8_t *w; const float *ws; const int8_t *x; const float *xs; float out[NCH]; int reps; } OPJ;
static void op_chunk(void *c, size_t i) {
    OPJ *j = c; float s = 0;
    for (int r = 0; r < j->reps; r++) s += dot_q8q8(j->w + (r & 63) * 4096, j->ws, j->x, j->xs, 4096);
    j->out[i] = s;
}
static void bench_machine(void) {
    size_t n = (size_t)256 << 17;                  /* 256 MB of uint64 */
    uint64_t *buf = malloc(n * 8); if (!buf) return;
    for (size_t i = 0; i < n; i++) buf[i] = i;
    BWJ j = {buf, n, {0}}; parfor(&j, bw_chunk);
    double t0 = now(); for (int r = 0; r < 4; r++) parfor(&j, bw_chunk); double dt = now() - t0;
    BW_GBS = 4.0 * n * 8 / dt / 1e9; SINK ^= (uint8_t)j.sums[0]; free(buf);
    int8_t *w = xcalloc(64 * 4096 + 4096, 1), *x = xcalloc(4096, 1); float *ws = xcalloc(128, 4), *xs = xcalloc(128, 4);
    for (int i = 0; i < 64 * 4096; i++) w[i] = (int8_t)(i * 7 % 255 - 127);
    for (int i = 0; i < 4096; i++) x[i] = (int8_t)(i * 13 % 255 - 127);
    for (int i = 0; i < 128; i++) ws[i] = xs[i] = 0.01f;
    OPJ o = {w, ws, x, xs, {0}, 400};
    t0 = now(); parfor(&o, op_chunk); dt = now() - t0;
    OPS_G = (double)NCH * 400 * 4096 / dt / 1e9;   /* billion int8 multiply-adds per second */
    SINK ^= (uint8_t)o.out[0];
    free(w); free(x); free(ws); free(xs);
}

static char TRAINDIR[8300]; static char LASTQ[2048];
static void tw_status(const char *json) {
    char p[8400], tp[8500]; snprintf(p, sizeof p, "%s/status.json", TRAINDIR); snprintf(tp, sizeof tp, "%s.part", p);
    FILE *f = fopen(tp, "w"); if (!f) return; fputs(json, f); fclose(f); rename(tp, p);
}
/* the window writes train/control as key=value lines; read every round so changes apply live */
typedef struct { int state; double limit, target; int roundlen, mode; char feeds[16][1024]; int nfeeds; } Ctl;   /* mode 0 auto, 1 turbo, 2 precise */
/* practice questions written from your fed files (train/prompts.txt), mixed with the built-in ones */
static void pick_prompt(int r, char *out, size_t n) {
    char p[8400]; snprintf(p, sizeof p, "%s/prompts.txt", TRAINDIR);
    if (r % 3 != 0) {
        FILE *f = fopen(p, "r");
        if (f) {
            static char line[2048]; long cnt = 0;
            while (fgets(line, sizeof line, f)) if (line[0] > ' ') cnt++;
            if (cnt > 0) {
                long want = ((long)r * 7919L) % cnt, i = 0; rewind(f);
                while (fgets(line, sizeof line, f)) if (line[0] > ' ' && i++ == want) break;
                fclose(f); line[strcspn(line, "\r\n")] = 0;
                snprintf(out, n, "%s", line);
                return;
            }
            fclose(f);
        }
    }
    practice_prompt(r, out, n);
}
static Ctl tw_control(void) {
    Ctl c; memset(&c, 0, sizeof c);
    char p[8400], line[1200]; snprintf(p, sizeof p, "%s/control", TRAINDIR);
    FILE *f = fopen(p, "r"); if (!f) return c;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strncmp(line, "stop", 4) || !strcmp(line, "state=stop")) c.state = 2;
        else if (!strncmp(line, "pause", 5) || !strcmp(line, "state=pause")) c.state = 1;
        else if (!strncmp(line, "limit=", 6)) c.limit = atof(line + 6);
        else if (!strncmp(line, "target=", 7)) c.target = atof(line + 7);
        else if (!strncmp(line, "round=", 6)) c.roundlen = atoi(line + 6);
        else if (!strncmp(line, "mode=", 5)) c.mode = !strcmp(line + 5, "turbo") ? 1 : !strcmp(line + 5, "precise") ? 2 : 0;
        else if (!strncmp(line, "feed=", 5) && c.nfeeds < 16) snprintf(c.feeds[c.nfeeds++], 1024, "%.1000s", line + 5);
    }
    fclose(f);
    return c;
}
/* feeding: pour a text file straight into the guessing memory - seconds, no model time */
static char FED[64][1024]; static int NFED; static long FED_WORDS;
static int feed_file(const char *path, char *note, size_t nn) {
    for (int i = 0; i < NFED; i++) if (!strcmp(FED[i], path)) return 0;
    if (NFED < 64) snprintf(FED[NFED++], 1024, "%s", path);
    FILE *f = fopen(path, "rb"); if (!f) { snprintf(note, nn, "could not open %s", path); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n > (64L << 20)) n = 64L << 20;
    char *txt = xcalloc((size_t)n + 1, 1); size_t got = fread(txt, 1, (size_t)n, f); fclose(f); txt[got] = 0;
    long before = NCORP, added = 0; size_t pos = 0;
    int *ids = xcalloc(70000, sizeof(int));
    while (pos < got) {                                   /* tokenize in ~64 KB pieces, cut at a space */
        size_t end = pos + 65536 < got ? pos + 65536 : got;
        while (end < got && end > pos + 1024 && txt[end] != ' ' && txt[end] != '\n') end--;
        char save = txt[end]; txt[end] = 0;
        int m = encode(txt + pos, ids, 70000);
        for (int i = 0; i < m; i++) corp_add(ids[i]);
        added += m;
        txt[end] = save; pos = end;
    }
    free(ids); free(txt); LAST[0] = LAST[1] = LAST[2] = -1;
    FED_WORDS += added; (void)before;
    const char *b = strrchr(path, '/');
    snprintf(note, nn, "fed %s: %ld words into the guessing memory%s", b ? b + 1 : path, NCORP - before, NCORP >= CAPCORP ? " (memory full)" : "");
    corp_save();
    return 1;
}
static void json_str(char *o, size_t n, const char *s) {   /* JSON string, only valid UTF-8 gets through */
    size_t k = 0; o[k++] = '"';
    const unsigned char *u = (const unsigned char *)s;
    while (*u && k + 12 < n) {
        unsigned char c = *u; int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        int ok = len > 0;
        for (int i = 1; ok && i < len; i++) if ((u[i] & 0xC0) != 0x80) ok = 0;
        if (!ok) { o[k++] = '?'; u++; continue; }
        if (len == 1) {
            if (c == '"' || c == '\\') { o[k++] = '\\'; o[k++] = (char)c; }
            else if (c == '\n') { o[k++] = '\\'; o[k++] = 'n'; }
            else if (c < 32) o[k++] = ' ';
            else o[k++] = (char)c;
        } else for (int i = 0; i < len; i++) o[k++] = (char)u[i];
        u += len;
    }
    o[k++] = '"'; o[k] = 0;
}

static double LIVE_PREV_BEST;
static void live_write(int tokens, double secs) {
    char p[8400], tp[8500], esc[3000], buf[3400];
    size_t L = strlen(LASTTEXT); json_str(esc, sizeof esc, L > 600 ? LASTTEXT + L - 600 : LASTTEXT);
    snprintf(buf, sizeof buf, "{\"tokens\":%d,\"secs\":%.2f,\"tps\":%.3f,\"limit\":%d,\"accept\":%.3f,\"text\":%s}",
             tokens, secs, secs > 0 ? tokens / secs : 0, MAXGEN, N_DRAFT ? (double)N_ACCEPT / N_DRAFT : 0, esc);
    snprintf(p, sizeof p, "%s/live.json", TRAINDIR); snprintf(tp, sizeof tp, "%s.part", p);
    FILE *f = fopen(tp, "w"); if (!f) return; fputs(buf, f); fclose(f); rename(tp, p);
    (void)LIVE_PREV_BEST;
}

typedef struct { int k; float mw; } Setting;
#define HIST 400

static void train(double seconds) {
    snprintf(TRAINDIR, sizeof TRAINDIR, "%s/train", PLATE); mkdir(TRAINDIR, 0755);
    ECHO = 0; MAXGEN = 64; LIVE_HOOK = live_write;
    char *js = xcalloc(1 << 16, 1), esc[4096];
    snprintf(js, 1 << 16, "{\"phase\":\"measuring your Mac\"}"); tw_status(js);
    bench_machine();
    /* bytes and math for one word */
    double exp_bytes = (double)(LY[0].gu.mat_bytes * 2 + LY[0].dn.mat_bytes);
    double dense_bytes = Q8_BYTES, word_bytes = dense_bytes + (double)TOPK * NL * exp_bytes;
    double word_ops = Q8_BYTES / 1.125 + (double)TOPK * NL * 3.0 * MOEI * DIM / 4.0;   /* lookup table: ~4 weights per op */
    double ceil_mem = BW_GBS * 1e9 / word_bytes, ceil_cpu = OPS_G * 1e9 / word_ops;
    double ceil_guess = 0;
    for (int k = 1; k <= MAXK; k++) {                           /* perfect guesses: k+1 words per pass */
        double pass_bytes = dense_bytes + (double)(k + 1) * TOPK * NL * exp_bytes;
        double wps = (k + 1) * fmin(BW_GBS * 1e9 / pass_bytes, OPS_G * 1e9 / (word_ops * (k + 1) * 0.35 + word_ops * 0.65));
        if (wps > ceil_guess) ceil_guess = wps;
    }
    float mw0 = MINW; double ppl0 = 0;
    char bp[8400]; snprintf(bp, sizeof bp, "%s/baseline.txt", TRAINDIR);
    { FILE *f = fopen(bp, "r"); if (f) { if (fscanf(f, "%lf", &ppl0) != 1) ppl0 = 0; fclose(f); } }   /* measured once, reused every run */
    if (ppl0 <= 0) {
        snprintf(js, 1 << 16, "{\"phase\":\"quality baseline (first run only)\"}"); tw_status(js);
        MINW = 0; ppl0 = perplexity(HOLDOUT); MINW = mw0;
        FILE *f = fopen(bp, "w"); if (f) { fprintf(f, "%.6f\n", ppl0); fclose(f); }
    }
    double ppl_best = MINW > 0 ? perplexity(HOLDOUT) : ppl0;

    Setting best = {DRAFTK, MINW}, cand = best;
    double hist_best[HIST], hist_cand[HIST]; int nh = 0;
    double bavg[40] = {0}, bfirst[40] = {0}, top = 0; int bn[40] = {0}, round = 0, kept = 0, tried = 0;   /* per answer length: fair comparisons */
    double best_avg = 0, first = 0; int nbest = 0;
    double t_start = now(), paused = 0;
    long hit0 = 0, probe0 = 0;
    char note[256] = "warming up";
    int rlen = 64, mode = 0; double target = 0;
    for (;;) {
        Ctl c = tw_control();
        if (c.state == 2) break;
        for (int i = 0; i < c.nfeeds; i++) feed_file(c.feeds[i], note, sizeof note);
        if (c.state == 1) { double tp = now(); sleep(1); paused += now() - tp; continue; }
        if (c.limit > 0 || seconds <= 0) seconds = c.limit;
        target = c.target; mode = c.mode;
        double el = now() - t_start - paused;
        if (seconds > 0 && el > seconds) { snprintf(note, sizeof note, "time is up - saved"); break; }
        if (target > 0 && best_avg >= target && nbest > 2) { snprintf(note, sizeof note, "target %.2f words/s reached - saved", target); break; }
        int early = seconds > 0 ? el < 0.35 * seconds : tried < 16;
        rlen = c.roundlen > 0 ? c.roundlen : mode == 1 ? 24 : mode == 2 ? 96 : early ? 24 : 64;
        MAXGEN = rlen;
        double margin = rlen <= 32 ? 1.06 : rlen >= 96 ? 1.03 : 1.04;
        int B = rlen / 8; if (B > 39) B = 39;
        best_avg = bavg[B]; first = bfirst[B]; nbest = bn[B];
        int trial = round % 2 == 1 && round > 2 && nbest >= 2;   /* odd rounds try something new, once this length has a baseline */
        if (trial) {
            cand = best;
            if (tried % 3 != 2) { int steps[] = {-2, -1, 1, 2}; cand.k = best.k + steps[(int)(rnd() * 4) & 3]; if (cand.k < 0) cand.k = 0; if (cand.k > MAXK) cand.k = MAXK; }
            else { float steps[] = {-0.04f, -0.02f, 0.02f, 0.04f}; cand.mw = best.mw + steps[(int)(rnd() * 4) & 3]; if (cand.mw < 0) cand.mw = 0; if (cand.mw > 0.12f) cand.mw = 0.12f; }
            tried++;
            if (cand.k == best.k && fabsf(cand.mw - best.mw) < 1e-4f) { round++; continue; }
            if (fabsf(cand.mw - best.mw) > 1e-4f) {           /* quality gate before speed */
                MINW = cand.mw; double pp = perplexity(HOLDOUT); MINW = best.mw;
                if (pp > ppl0 * 1.02) { snprintf(note, sizeof note, "skip-share %.2f rejected: quality dropped %.1f%%", cand.mw, 100 * (pp / ppl0 - 1)); round++; continue; }
            }
        }
        Setting use = trial ? cand : best;
        DRAFTK = use.k; MINW = use.mw;
        state_reset(); seen_clear(); PENDING_END = 0; LAST[0] = LAST[1] = LAST[2] = -1;
        long g0 = N_GEN; double tg0 = T_GEN; long h0 = N_HIT, p0 = N_PROBE, a0 = N_ACCEPT, d0 = N_DRAFT, ps0 = N_PASS;
        char pq[2048]; pick_prompt(round, pq, sizeof pq); answer(pq);
        snprintf(LASTQ, sizeof LASTQ, "%s", pq);
        double wps = (T_GEN - tg0) > 0 ? (N_GEN - g0) / (T_GEN - tg0) : 0;
        double acc = N_DRAFT - d0 ? (double)(N_ACCEPT - a0) / (N_DRAFT - d0) : 0, wpp = N_PASS - ps0 ? (double)(N_GEN - g0) / (N_PASS - ps0) : 1;
        double hits = N_PROBE - p0 ? (double)(N_HIT - h0) / (N_PROBE - p0) : 0;
        hit0 = N_HIT - h0; probe0 = N_PROBE - p0; (void)hit0; (void)probe0;
        if (!trial) {
            best_avg = nbest ? 0.6 * best_avg + 0.4 * wps : wps; nbest++;
            if (!first && nbest >= 2) first = best_avg;
            if (nh < HIST) { hist_best[nh] = wps; hist_cand[nh] = -1; nh++; }
        } else {
            if (nh < HIST) { hist_best[nh] = -1; hist_cand[nh] = wps; nh++; }
            if (wps > best_avg * margin) {
                kept++; snprintf(note, sizeof note, "kept: guess %d ahead, skip-share %.2f  (%.2f vs %.2f tok/s)", cand.k, cand.mw, wps, best_avg);
                best = cand; best_avg = 0.5 * best_avg + 0.5 * wps;
                tuned_save(best.k, best.mw);
                if (fabsf(best.mw - mw0) > 1e-4f) { MINW = best.mw; ppl_best = perplexity(HOLDOUT); mw0 = best.mw; }
            } else snprintf(note, sizeof note, "tried guess %d ahead, skip-share %.2f: %.2f tok/s, not better", cand.k, cand.mw, wps);
        }
        bavg[B] = best_avg; bfirst[B] = first; bn[B] = nbest;
        if (best_avg > top) top = best_avg;
        if (round % 4 == 3) save_all();
        /* status for the window */
        long used = 0; for (long i = 0; i < (long)NL * NE; i++) if (HEAT[i] || HEAT_ALL[i]) used++;
        char hb[6000] = "", hc[6000] = ""; size_t lb = 0, lc = 0;
        int from = nh > 120 ? nh - 120 : 0;
        for (int i = from; i < nh; i++) {
            lb += (size_t)snprintf(hb + lb, sizeof hb - lb, "%s%.3f", i > from ? "," : "", hist_best[i]);
            lc += (size_t)snprintf(hc + lc, sizeof hc - lc, "%s%.3f", i > from ? "," : "", hist_cand[i]);
            if (lb > sizeof hb - 32 || lc > sizeof hc - 32) break;
        }
        { size_t L = strlen(LASTTEXT); char tmp[2400]; snprintf(tmp, sizeof tmp, "Q: %.300s\n\n%s", LASTQ, L > 2000 ? LASTTEXT + L - 2000 : LASTTEXT); json_str(esc, sizeof esc, tmp); }
        char nesc[600]; json_str(nesc, sizeof nesc, note);
        snprintf(js, 1 << 16,
            "{\"phase\":\"training\",\"round\":%d,\"elapsed\":%.0f,\"limit\":%.0f,"
            "\"wps_now\":%.3f,\"wps_best\":%.3f,\"wps_first\":%.3f,\"wps_top\":%.3f,"
            "\"accept\":%.3f,\"wpp\":%.3f,\"hits\":%.3f,"
            "\"draft_k\":%d,\"min_share\":%.3f,\"trial\":%d,"
            "\"ppl0\":%.4f,\"ppl\":%.4f,\"corpus\":%ld,\"woken\":%ld,\"experts\":%d,"
            "\"bw\":%.2f,\"ops\":%.2f,\"ceil_mem\":%.3f,\"ceil_cpu\":%.3f,\"ceil_guess\":%.3f,"
            "\"kept\":%d,\"tried\":%d,\"note\":%s,\"text\":%s,\"hb\":[%s],\"hc\":[%s],"
            "\"mode\":\"%s\",\"round_len\":%d,\"target\":%.2f,\"fed\":%ld}",
            round, now() - t_start - paused, seconds, wps, best_avg, first, top, acc, wpp, hits,
            best.k, best.mw, trial, ppl0, ppl_best, NCORP, used, NL * NE,
            BW_GBS, OPS_G, ceil_mem, ceil_cpu, ceil_guess, kept, tried, nesc, esc, hb, hc,
            mode == 1 ? "turbo" : mode == 2 ? "precise" : "auto", rlen, target, FED_WORDS);
        tw_status(js);
        round++;
    }
    DRAFTK = best.k; MINW = best.mw; tuned_save(best.k, best.mw); save_all();
    { char nesc[600]; json_str(nesc, sizeof nesc, note);
      snprintf(js, 1 << 16, "{\"phase\":\"done\",\"wps_best\":%.3f,\"wps_first\":%.3f,\"draft_k\":%d,\"min_share\":%.3f,\"round\":%d,\"ceil_mem\":%.3f,\"ceil_cpu\":%.3f,\"ceil_guess\":%.3f,\"corpus\":%ld,\"fed\":%ld,\"note\":%s}",
             best_avg, first, best.k, best.mw, round, ceil_mem, ceil_cpu, ceil_guess, NCORP, FED_WORDS, nesc); }
    tw_status(js);
    free(js);
}

/* ------------------------------------------------------------------ main */

static int BATCHTEST = 0;
static void on_sigint(int s) { (void)s; save_all(); printf("\n"); _exit(0); }

static int parse_ids(const char *s, int *ids, int max) {
    int n = 0; while (*s && n < max) { char *e; long v = strtol(s, &e, 10); if (e == s) { s++; continue; } ids[n++] = (int)v; s = e; }
    return n;
}

int main(int argc, char **argv) {
    const char *home = getenv("HOME"), *prompt = NULL, *tokenize = NULL, *dump = NULL, *dump_out = NULL, *ids_s = NULL, *pplf = NULL;
    int set_k = -1, set_lines = -1, render_test = 0; const char *set_len = NULL; float set_mw = -1; double train_s = -1;
    char root[2048]; const char *hh = getenv("HOLOMOE_HOME");
    if (hh && *hh) snprintf(root, sizeof root, "%s", hh); else snprintf(root, sizeof root, "%s/holomoe", home ? home : ".");
    snprintf(PLATE, sizeof PLATE, "%s/plate", root);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { help(); return 0; }
        if (!strcmp(a, "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(a, "--model") && i + 1 < argc) {          /* 35b (default) | 30b | 2b: each model has its own plate */
            const char *m = argv[++i];
            if (!strcmp(m, "35b")) snprintf(PLATE, sizeof PLATE, "%s/plate", root); else snprintf(PLATE, sizeof PLATE, "%s/plate-%s", root, m);
            struct stat sb; if (stat(PLATE, &sb)) { fprintf(stderr, "holomoe: no model '%s' yet - get it with: python3 shardpress.py all --model %s\n", m, m); return 1; }
        }
        else if (!strcmp(a, "--think")) THINKING = 1;
        else if (!strcmp(a, "--float-act")) FLOAT_ACT = 1;
        else if (!strcmp(a, "--bf16")) Q8 = 0;
        else if (!strcmp(a, "--no-simd")) SIMD = 0;
        else if (!strcmp(a, "--length") && i + 1 < argc) set_len = argv[++i];
        else if (!strcmp(a, "--repeat-penalty") && i + 1 < argc) PRES = (float)atof(argv[++i]);
        else if (!strcmp(a, "--plain")) COLOR = 0;
        else if (!strcmp(a, "--lean")) LEAN = 1;
        else if (!strcmp(a, "--full-vocab")) TRIM = 0;
        else if (!strcmp(a, "--raw")) RENDER = 0;
        else if (!strcmp(a, "--render-test")) render_test = 1;
        else if (!strcmp(a, "--lines") && i + 1 < argc) set_lines = atoi(argv[++i]);
        else if (!strcmp(a, "--no-predict")) PREDICT = 0;
        else if (!strcmp(a, "--preload") && i + 1 < argc) PRELOAD_GB = atof(argv[++i]);
        else if (!strcmp(a, "--guess") && i + 1 < argc) set_k = atoi(argv[++i]);
        else if (!strcmp(a, "--skip") && i + 1 < argc) set_mw = (float)atof(argv[++i]);
        else if (!strcmp(a, "--no-learn")) NO_LEARN = 1;
        else if (!strcmp(a, "--train") && i + 1 < argc) train_s = atof(argv[++i]);
        else if (!strcmp(a, "--ppl") && i + 1 < argc) pplf = argv[++i];
        else if (!strcmp(a, "--batch")) BATCHTEST = 1;
        else if (!strcmp(a, "--ctx") && i + 1 < argc) CTX = atoi(argv[++i]);
        else if (!strcmp(a, "--temp") && i + 1 < argc) TEMP = (float)atof(argv[++i]);
        else if (!strcmp(a, "--seed") && i + 1 < argc) RNG ^= (uint64_t)atoll(argv[++i]) * 0x2545F4914F6CDD1DULL;
        else if (!strcmp(a, "-n") && i + 1 < argc) MAXGEN = atoi(argv[++i]);
        else if (!strcmp(a, "--tokenize") && i + 1 < argc) tokenize = argv[++i];
        else if (!strcmp(a, "--ids") && i + 1 < argc) ids_s = argv[++i];
        else if (!strcmp(a, "--dump") && i + 2 < argc) { dump = argv[++i]; dump_out = argv[++i]; }
        else if (a[0] != '-') snprintf(PLATE, sizeof PLATE, "%s", a);
        else { fprintf(stderr, "unknown option %s\n", a); return 1; }
    }
    if (!isatty(1) || getenv("NO_COLOR")) COLOR = 0;
    if (render_test) {                                  /* format text from stdin the way answers are shown */
        static char buf[1 << 16]; size_t n = fread(buf, 1, sizeof buf - 1, stdin); buf[n] = 0; render_text(buf); return 0;
    }
    float keep_temp = TEMP; set_think(THINKING); if (keep_temp != 0.7f) TEMP = keep_temp;
    if (CTX < 64) CTX = 64;
    char path[8192];
    snprintf(path, sizeof path, "%s/meta/model.cfg", PLATE); cfg_load(path);
    if (THINK < 0) THINKING = 0;                        /* e.g. Qwen3-2507 Instruct: no thinking mode */
    snprintf(path, sizeof path, "%s/meta/vocab.bin", PLATE); vocab_load(path);
    if (tokenize) {
        int max = (int)strlen(tokenize) + 8, *ids = xcalloc((size_t)max, sizeof(int)), n = encode(tokenize, ids, max);
        for (int i = 0; i < n; i++) printf("%d%c", ids[i], i + 1 < n ? ' ' : '\n');
        return 0;
    }
    double t0 = now();
    model_load(); scratch_alloc();
    int chat = !dump && !ids_s && !pplf;
    if (chat) {
        len_load();
        heat_load(); preload(); tuned_load(); corp_load(); if (train_s < 0) { lt_load(); search_load(); }
    }
    if (set_len) len_set(set_len);
    if (set_lines >= 0) LINE_CAP = set_lines;
    if (set_k >= 0) DRAFTK = set_k > MAXK ? MAXK : set_k;
    if (set_mw >= 0) MINW = set_mw;
    BOOT_SECS = now() - t0;
    if (NE) fprintf(stderr, "[ready in %.1fs: %.2f GB attention + %.2f GB of your hottest experts in RAM, the rest stream from disk]\n", now() - t0, Q8_BYTES / 1e9, PRELOAD_BYTES / 1e9);
    else fprintf(stderr, "[ready in %.1fs: %.2f GB of weights in RAM (dense model)]\n", now() - t0, Q8_BYTES / 1e9);
    if (dump) {                                         /* verification: logits for every position */
        int ids[4096], n = parse_ids(dump, ids, 4096);
        FILE *f = fopen(dump_out, "wb"); if (!f) die("cannot write dump");
        if (BATCHTEST) { forward_batch(ids, n - 1, 0); forward(ids[n - 1], 1); fwrite(LOG, 4, (size_t)VOCAB, f); }
        else for (int i = 0; i < n; i++) { forward(ids[i], 1); fwrite(LOG, 4, (size_t)VOCAB, f); }
        fclose(f); fprintf(stderr, "[dumped %d x %d logits]\n", BATCHTEST ? 1 : n, VOCAB);
        return 0;
    }
    if (ids_s) {                                        /* top 5 next tokens after raw ids */
        int ids[4096], n = parse_ids(ids_s, ids, 4096);
        forward_batch(ids, n, 1);
        for (int r = 0; r < 5; r++) {
            int b = -1; for (int i = 0; i < VOCAB; i++) if ((b < 0 || LOG[i] > LOG[b])) b = i;
            printf("%8d %10.4f  '", b, LOG[b]); put_tok(b, 1); printf("'\n"); LOG[b] = -1e30f;
        }
        return 0;
    }
    if (pplf) {
        FILE *f = fopen(pplf, "rb"); if (!f) die("cannot read text file");
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *txt = xcalloc((size_t)n + 1, 1); if (fread(txt, 1, (size_t)n, f) != (size_t)n) die("read"); fclose(f);
        printf("perplexity %.4f\n", perplexity(txt));
        return 0;
    }
    signal(SIGINT, on_sigint);
    if (train_s >= 0) { train(train_s); return 0; }
    if (prompt) { chat_answer(prompt); if (getenv("HOLOMOE_STATS")) stats(); save_all(); return 0; }
    repl();
    save_all();
    return 0;
}
