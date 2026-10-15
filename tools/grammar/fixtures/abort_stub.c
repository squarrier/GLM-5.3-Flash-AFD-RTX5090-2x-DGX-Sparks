/* grammar_abi refusal fixture (Claude Opus 5.5 via Hermes): exports the release libtfgrammar symbols EXCEPT
 * tfg_drafts, every body abort(). A loader that calls any foreign function before all 12 symbols resolve crashes. */
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

void *tfg_open(const char *a, size_t b, int c, const int32_t *d, int e, char *f, size_t g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; abort(); }
void tfg_close(void *c) { (void)c; abort(); }
int tfg_words(void *c) { (void)c; abort(); }
void *tfg_compile(void *c, int k, const char *t, size_t n, char *e, size_t l) { (void)c; (void)k; (void)t; (void)n; (void)e; (void)l; abort(); }
void tfg_free(void *g) { (void)g; abort(); }
void *tfg_matcher(void *g) { (void)g; abort(); }
void tfg_matcher_free(void *m) { (void)m; abort(); }
int tfg_accept(void *m, int32_t t) { (void)m; (void)t; abort(); }
int tfg_rollback(void *m, int n) { (void)m; (void)n; abort(); }
int tfg_terminated(void *m) { (void)m; abort(); }
int tfg_fill(void *m, int32_t *w, int n) { (void)m; (void)w; (void)n; abort(); }
