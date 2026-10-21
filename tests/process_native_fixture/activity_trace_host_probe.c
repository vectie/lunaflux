#include "activity_trace.c"
#include <assert.h>

/* CPU-only ownership/format boundaries of the exact native collector. GPU
 * coverage is a separate approved-child fixture, never inferred from this. */
int main(void) {
  uint8_t *p[BUFFERS], *extra;
  size_t size, max;
  for (int i=0;i<BUFFERS;++i) {
    request_buffer(&p[i],&size,&max);
    assert(p[i] == storage[i] && size == BUFFER_BYTES && max == 0);
  }
  request_buffer(&extra,&size,&max);
  assert(extra == NULL && size == 0 && atomic_load(&refused) == 1);
  atomic_store(&occupied[3],0);
  request_buffer(&extra,&size,&max);
  assert(extra == storage[3]);
  records = tmpfile(); assert(records);
  windows=2; begins[0]=100;ends[0]=200;begins[1]=300;ends[1]=400;
  assert(window_for(100,110)==1 && window_for(199,220)==1);
  assert(window_for(200,210)==0 && window_for(300,310)==2);
  assert(window_for(0,110)==0 && invalid==1);
  emit("K",100,120,7,9,11,13,"fixture",0,0,1,2,3,4,5,6);
  emit("K",300,320,7,9,11,13,"bad\tname",0,0,1,2,3,4,5,6);
  assert(count==1 && invalid==2);
  rewind(records); char line[512]; assert(fgets(line,sizeof(line),records));
  assert(!strcmp(line,"1\tK\t100\t120\t7\t9\t11\t13\t0\t0\t1\t2\t3\t4\t5\t6\tfixture\n"));
  assert(fgetc(records)==EOF);
  fclose(records); records=NULL;
  puts("bounded buffers, exhaustion, reuse, timestamp windows, malformed names and exact schema pass");
}
