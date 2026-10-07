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
  assert(selected_decode_capture_bound(1,32)==1);
  assert(selected_decode_capture_bound(2,32)==8);
  assert(selected_decode_capture_bound(4,32)==8);
  assert(selected_decode_capture_bound(8,32)==8);
  assert(selected_decode_capture_bound(9,32)==16);
  assert(selected_decode_capture_bound(17,32)==32);
  assert(selected_decode_capture_bound(9,12)==12);
  assert(selected_decode_capture_bound(9,12,12)==12);
  assert(selected_merge_block(64)==64);
  assert(selected_merge_block(128,64)==64);
  assert(selected_merge_block(64,128)==128);
  assert(selected_mixed_prefill_rows(8)==7);
  assert(selected_mixed_prefill_rows(8,1)==1);
  assert(selected_mixed_prefill_rows(16,8)==8);
  auto work=selected_row_work("2041:4090,1:8194,1:8198",3,2043,1,16,8192);
  assert(work.size()==3 && work[0].first==2041 && work[2].second==8198);
  auto two_prefill=selected_row_work("79:8113,1963:0,1:8211,1:8207,1:8203,1:8199,1:8195,1:8216",8,2048,2,8,16384);
  assert(two_prefill[0].first==79 && two_prefill[1].first==1963);
  for(const char* invalid : {"2:0,1:4,", "2:0,1:-1", "2:0,2:4", "2:0", "2:0,1:4x", "2:0,1:2147483647", "2:0,1:9223372036854775807"}) {
    bool rejected=false;
    try { (void)selected_row_work(invalid,2,3,1,16,8192); }
    catch(const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
  assert(selected_bucket_tokens(3,32)==4); // mixed capture: unlike decode's minimum eight
  for (auto invalid : {std::pair<int,int>{1,0},{8,-1},{8,8},{8,9}}) {
    bool rejected=false;
    try { (void)selected_mixed_prefill_rows(invalid.first,invalid.second); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
  for (auto invalid : {std::pair<int,int>{0,64},{128,-1},{128,1025},{1025,64}}) {
    bool rejected=false;
    try { (void)selected_merge_block(invalid.first,invalid.second); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
  for (int invalid : {1,2,4,64}) {
    bool rejected=false;
    try { (void)selected_decode_capture_bound(2,32,invalid); }
    catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
  }
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
