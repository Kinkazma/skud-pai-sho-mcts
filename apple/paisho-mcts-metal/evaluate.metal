#include <metal_stdlib>
using namespace metal;
// PMG1: row-major 17x17, 0 empty; 1 + kind.index + 12 * owner.index.
bool playable(int r, int c) { return abs(r-8)+abs(c-8)<=12; }
bool gate(int r,int c) { return (r==8 && (c==0||c==16)) || (c==8 && (r==0||r==16)); }
int kind(uchar t) { return (int(t)-1)%12; }
int owner(uchar t) { return (int(t)-1)/12; }
bool knot(threadgroup const uchar *b,int r,int c) {
    for(int dr=-1;dr<=1;dr++) for(int dc=-1;dc<=1;dc++) {
        int rr=r+dr,cc=c+dc;
        if((dr||dc)&&rr>=0&&rr<17&&cc>=0&&cc<17&&playable(rr,cc)) {
            uchar t=b[rr*17+cc]; if(t && kind(t)==10) return true;
        }
    }
    return false;
}
int harmony_owner(uchar a,uchar b) {
    int ka=kind(a),kb=kind(b);
    if(ka<6 && kb<6 && owner(a)==owner(b) && (abs(ka-kb)==1||abs(ka-kb)==5)) return owner(a);
    if(ka==6 && kb<6) return owner(b);
    if(ka<6 && kb==6) return owner(a);
    return -1;
}
kernel void features(device const uchar *input [[buffer(0)]], device int4 *output [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup uchar board[289];
    threadgroup int4 totals[32];
    for(uint i=lane;i<289;i+=32) board[i]=input[group*289+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int4 result(0);
    for(int i=int(lane);i<289;i+=32) {
        uchar a=board[i]; if(!a) continue;
        int r=i/17,c=i%17;
        if(kind(a)<8) { int s=owner(a)==0 ? 1:-1; result.w+=s; if(!gate(r,c)) result.z+=s; }
        if(gate(r,c)) continue;
        for(int dir=0;dir<2;dir++) {
            int rr=r,cc=c;
            while(true) {
                if(dir==0) cc++; else rr++;
                if(rr>=17||cc>=17||!playable(rr,cc)||gate(rr,cc)) break;
                uchar b=board[rr*17+cc]; if(!b) continue;
                int o=harmony_owner(a,b);
                if(o>=0 && !knot(board,r,c) && !knot(board,rr,cc)) {
                    bool rock=false;
                    for(int k=0;k<17;k++) { uchar t=board[dir==0 ? r*17+k : k*17+c]; if(t && kind(t)==8) rock=true; }
                    if(!rock) {
                        int s=o==0 ? 1:-1; result.x+=s;
                        if(dir==0 ? (c<8&&cc>8&&r!=8) : (r<8&&rr>8&&c!=8)) result.y+=s;
                    }
                }
                break;
            }
        }
    }
    totals[lane]=result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) { int4 sum(0); for(int k=0;k<32;k++) sum+=totals[k]; output[group]=sum; }
}
