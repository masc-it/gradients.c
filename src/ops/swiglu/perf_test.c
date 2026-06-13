#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <gradients/gradients.h>
#include <mach/mach_time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static double now_s(void){ static mach_timebase_info_data_t info; static double scale=0.0; if(scale==0.0){ mach_timebase_info(&info); scale=((double)info.numer/(double)info.denom)*1e-9;} return (double)mach_absolute_time()*scale; }
static size_t align_up(size_t v,size_t a){ return (v+a-1u)&~(a-1u); }
static bool ok(gd_context*ctx, gd_status st, const char*e){ if(st==GD_OK) return true; fprintf(stderr,"FAIL %s -> %s ctx=%s\n",e,gd_status_string(st),ctx?gd_context_error(ctx):"none"); return false; }
#define OK(expr) do{ if(!ok(ctx,(expr),#expr)) return 1; }while(0)
static int run_fwd(gd_context*ctx, gd_tensor*x12){ gd_tensor y; OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty())); OK(gd_swiglu_split(ctx,x12,&y)); OK(gd_end_step(ctx)); OK(gd_synchronize(ctx)); return 0; }
static int run_bwd(gd_context*ctx, gd_tensor*x12, gd_tensor*grad){ gd_tensor dx12; OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty())); OK(gd_swiglu_split_backward(ctx,x12,grad,&dx12)); OK(gd_end_step(ctx)); OK(gd_synchronize(ctx)); return 0; }
static int run_auto(gd_context*ctx, gd_tensor*x12, gd_tensor*grad){ gd_tensor y; x12->requires_grad=true; OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty())); OK(gd_swiglu_split(ctx,x12,&y)); OK(gd_backward(ctx,&y,grad)); OK(gd_end_step(ctx)); OK(gd_synchronize(ctx)); return 0; }
typedef int (*fn_t)(gd_context*, gd_tensor*, gd_tensor*);
static int measure(const char*name, gd_context*ctx, gd_tensor*x12, gd_tensor*grad, fn_t fn, int warmup, int iters, size_t elems){ for(int i=0;i<warmup;i++){ if(fn(ctx,x12,grad)!=0) return 1; } double t0=now_s(); for(int i=0;i<iters;i++){ if(fn(ctx,x12,grad)!=0) return 1; } double dt=(now_s()-t0)/(double)iters; printf("[SWIGLU_SPLIT_EXACT][%s] avg_ms=%.4f elems=%zu Gelem/s=%.2f\n", name, dt*1e3, elems, (double)elems*1e-9/dt); return 0; }
static int fwd_wrap(gd_context*ctx, gd_tensor*x12, gd_tensor*grad){ (void)grad; return run_fwd(ctx,x12); }
static int bwd_wrap(gd_context*ctx, gd_tensor*x12, gd_tensor*grad){ return run_bwd(ctx,x12,grad); }
static int auto_wrap(gd_context*ctx, gd_tensor*x12, gd_tensor*grad){ return run_auto(ctx,x12,grad); }
int main(void){ const int64_t rows=49152; const int64_t h=1024; const size_t x12_elems=(size_t)rows*(size_t)(2*h); const size_t y_elems=(size_t)rows*(size_t)h; const size_t x12_bytes=x12_elems*2u; const size_t y_bytes=y_elems*2u; gd_memory_config cfg; memset(&cfg,0,sizeof(cfg)); cfg.params_bytes=align_up(x12_bytes+y_bytes+64u*1024u*1024u,4096u); cfg.state_bytes=4u*1024u*1024u; cfg.scratch_slot_bytes=align_up(x12_bytes*4u+y_bytes*6u+256u*1024u*1024u,4096u); cfg.data_slot_bytes=4u*1024u*1024u; cfg.scratch_slots=3u; cfg.data_slots=2u; cfg.default_alignment=256u; gd_context*ctx=NULL; if(!ok(NULL,gd_context_create(&cfg,&ctx),"gd_context_create")) return 1; int64_t xshape[2]={rows,2*h}; int64_t yshape[2]={rows,h}; gd_tensor x12, grad; OK(gd_tensor_rand_uniform(ctx,GD_ARENA_PARAMS,GD_DTYPE_F16,gd_shape_make(2,xshape),256u,1234u,-4.0f,4.0f,&x12)); OK(gd_tensor_rand_uniform(ctx,GD_ARENA_PARAMS,GD_DTYPE_F16,gd_shape_make(2,yshape),256u,5678u,-0.5f,0.5f,&grad)); OK(gd_context_seal_params(ctx)); printf("[SWIGLU_SPLIT_EXACT] rows=%lld hidden=%lld x12_elems=%zu y_elems=%zu warmup=3 iters=10\n",(long long)rows,(long long)h,x12_elems,y_elems); if(measure("fwd",ctx,&x12,&grad,fwd_wrap,3,10,y_elems)!=0) return 1; if(measure("bwd",ctx,&x12,&grad,bwd_wrap,3,10,y_elems)!=0) return 1; if(measure("autograd",ctx,&x12,&grad,auto_wrap,3,10,y_elems)!=0) return 1; gd_context_destroy(ctx); return 0; }
