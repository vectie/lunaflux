#include "selected_policy_geometry.h"
#include <cassert>
int main() {
  assert(selected_bucket_tokens(129,2048)==256);
  assert(selected_bucket_tokens(1500,1536)==1536);
  assert(selected_grid_x(128,128,32,16,false,false)==8);
  assert(selected_grid_x(64,128,32,32,false,false)==4);
  assert(selected_grid_x(158,2048,32,64,true,false)==63);
  assert(selected_grid_x(158,8,32,64,true,false)==8);
  assert(selected_grid_x(32,8,32,1,false,true)==8);
  assert(selected_capture_bound(2,32)==2);
  assert(selected_capture_bound(2,32,8)==8);
  for (int invalid : {1,3,64}) {
    bool rejected=false;
    try { (void)selected_capture_bound(2,32,invalid); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
  for(int tile : {16,32,64,128}) for(int rows=1;rows<=32;rows++)
    for(int extra=0;extra<=256;extra++)
      assert(selected_grid_x(4096,rows+extra,rows,tile,true,false)==rows+extra/tile);
}
