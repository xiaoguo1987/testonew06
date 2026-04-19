// poker_engine.cpp — 高性能 C++ 扑克引擎 (pybind11)
// 替代 Python 的 cards.py / poker_game.py / hand_evaluator.py / state_representation.py
// 编译: python setup_engine.py build_ext --inplace

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cmath>
#include <string>

namespace py = pybind11;

// ================================================================
//  Card encoding:  code = rank_idx * 4 + suit_idx
//    rank_idx 0-12  →  2,3,4,...,K,A
//    suit_idx 0-3   →  ♥♦♣♠
//    rank_value = rank_idx + 2   (2–14, A=14)
// ================================================================
static inline int rank_of(int c)    { return c >> 2; }        // c/4
static inline int suit_of(int c)    { return c & 3;  }        // c%4
static inline int rank_val(int c)   { return (c >> 2) + 2; }  // 2-14

// ================================================================
//  Hand evaluation
// ================================================================
enum HandType : int8_t {
    HIGH_CARD=0, ONE_PAIR=1, TWO_PAIR=2, THREE_KIND=3,
    STRAIGHT=4, FLUSH=5, FULL_HOUSE=6, FOUR_KIND=7,
    STRAIGHT_FLUSH=8, ROYAL_FLUSH=9
};

struct HandRank {
    int8_t type = -1;
    int8_t k[5] = {};   // kickers (desc)
    int     kn  = 0;
    int compare(const HandRank& o) const {
        if (type != o.type) return type > o.type ? 1 : -1;
        int n = std::min(kn, o.kn);
        for (int i = 0; i < n; i++)
            if (k[i] != o.k[i]) return k[i] > o.k[i] ? 1 : -1;
        return 0;
    }
};

static HandRank eval5(const int8_t c[5]) {
    HandRank r;
    int rc[13]={}, sc[4]={};
    int rv[5];
    for (int i=0;i<5;i++) { int ri=rank_of(c[i]); rc[ri]++; sc[suit_of(c[i])]++; rv[i]=ri+2; }
    std::sort(rv, rv+5, std::greater<int>());

    bool fl = false;
    for (int i=0;i<4;i++) if (sc[i]==5) { fl=true; break; }

    bool st = false; int sh = 0;
    // unique check
    bool uniq = true;
    for (int i=0;i<4;i++) if (rv[i]==rv[i+1]) { uniq=false; break; }
    if (uniq) {
        if (rv[0]-rv[4]==4) { st=true; sh=rv[0]; }
        else if (rv[0]==14&&rv[1]==5&&rv[2]==4&&rv[3]==3&&rv[4]==2) { st=true; sh=5; }
    }

    if (st && fl) {
        r.type = (sh==14) ? ROYAL_FLUSH : STRAIGHT_FLUSH;
        r.k[0]=sh; r.kn=1; return r;
    }
    // four kind
    for (int i=12;i>=0;i--) if (rc[i]==4) {
        r.type=FOUR_KIND; r.k[0]=i+2;
        for (int j=0;j<5;j++) if (rv[j]!=i+2) { r.k[1]=rv[j]; break; }
        r.kn=2; return r;
    }
    // full house
    int t3=-1,p2=-1;
    for (int i=12;i>=0;i--) { if (rc[i]==3&&t3<0) t3=i; if (rc[i]==2&&p2<0) p2=i; }
    if (t3>=0&&p2>=0) { r.type=FULL_HOUSE; r.k[0]=t3+2; r.k[1]=p2+2; r.kn=2; return r; }
    if (fl) { r.type=FLUSH; for(int i=0;i<5;i++) r.k[i]=rv[i]; r.kn=5; return r; }
    if (st) { r.type=STRAIGHT; r.k[0]=sh; r.kn=1; return r; }
    if (t3>=0) {
        r.type=THREE_KIND; r.k[0]=t3+2; int ki=1;
        for (int i=0;i<5&&ki<3;i++) if (rv[i]!=t3+2) r.k[ki++]=rv[i];
        r.kn=3; return r;
    }
    // pairs
    int pa[2]={-1,-1}; int np=0;
    for (int i=12;i>=0;i--) if (rc[i]==2&&np<2) pa[np++]=i;
    if (np==2) {
        r.type=TWO_PAIR; r.k[0]=pa[0]+2; r.k[1]=pa[1]+2;
        for (int i=0;i<5;i++) if (rv[i]!=pa[0]+2&&rv[i]!=pa[1]+2) { r.k[2]=rv[i]; break; }
        r.kn=3; return r;
    }
    if (np==1) {
        r.type=ONE_PAIR; r.k[0]=pa[0]+2; int ki=1;
        for (int i=0;i<5&&ki<4;i++) if (rv[i]!=pa[0]+2) r.k[ki++]=rv[i];
        r.kn=4; return r;
    }
    r.type=HIGH_CARD; for(int i=0;i<5;i++) r.k[i]=rv[i]; r.kn=5; return r;
}

// C(7,5) = 21 combos
static const int C75[21][5] = {
    {0,1,2,3,4},{0,1,2,3,5},{0,1,2,3,6},{0,1,2,4,5},{0,1,2,4,6},
    {0,1,2,5,6},{0,1,3,4,5},{0,1,3,4,6},{0,1,3,5,6},{0,1,4,5,6},
    {0,2,3,4,5},{0,2,3,4,6},{0,2,3,5,6},{0,2,4,5,6},{0,3,4,5,6},
    {1,2,3,4,5},{1,2,3,4,6},{1,2,3,5,6},{1,2,4,5,6},{1,3,4,5,6},
    {2,3,4,5,6}
};

static HandRank eval7(const int8_t cards[7]) {
    HandRank best; best.type=-1;
    int8_t cb[5];
    for (int c=0;c<21;c++) {
        for (int i=0;i<5;i++) cb[i]=cards[C75[c][i]];
        HandRank h = eval5(cb);
        if (h.compare(best)>0) best=h;
    }
    return best;
}

// ================================================================
//  Deck
// ================================================================
class Deck {
    int8_t cards_[52];
    int top_;
    std::mt19937 rng_;
public:
    Deck() : top_(52) {
        std::random_device rd; rng_.seed(rd());
        for (int i=0;i<52;i++) cards_[i]=(int8_t)i;
    }
    void shuffle() {
        for (int i=0;i<52;i++) cards_[i]=(int8_t)i;
        std::shuffle(cards_, cards_+52, rng_);
        top_=52;
    }
    int8_t deal() { return cards_[--top_]; }
};

// ================================================================
//  GameEngine  —  complete 6-max poker table
// ================================================================
class GameEngine {
    int sb_, bb_, init_stack_, np_;

    int32_t stacks_[6], bets_[6];
    bool    active_[6], allin_[6];
    int8_t  hole_[6][2];
    int     hcnt_[6];          // hole card count per player

    int8_t  board_[5];
    int     bcnt_;             // board card count
    int     pot_, cbet_;       // pot, current bet
    int     dpos_, cplayer_;   // dealer pos, current player idx
    int     round_, acnt_;     // round (1-4), action count
    std::vector<int> winners_;
    Deck    deck_;

    void post_blinds() {
        int sp=(dpos_+1)%np_, bp=(dpos_+2)%np_;
        auto bet=[&](int p, int amt){
            if (stacks_[p]>=amt) {
                bets_[p]+=amt; stacks_[p]-=amt; pot_+=amt;
                if (stacks_[p]==0) allin_[p]=true;
            }
        };
        bet(sp, sb_); cbet_=sb_;
        bet(bp, bb_); cbet_=bb_;
    }
    void deal_hole() {
        for (int c=0;c<2;c++)
            for (int p=0;p<np_;p++) { hole_[p][c]=deck_.deal(); hcnt_[p]=c+1; }
    }
    void setup_round() {
        if (round_!=1) cbet_=0;
        acnt_=0;
        cplayer_ = (round_==1) ? (dpos_+2)%np_ : dpos_;
    }
    void do_showdown() {
        std::vector<int> act;
        for (int i=0;i<np_;i++) if (active_[i]) act.push_back(i);
        if (act.empty()) return;
        winners_.clear(); winners_.push_back(act[0]);
        for (size_t i=1;i<act.size();i++) {
            int ch=act[i], bst=winners_[0];
            int8_t h1[7], h2[7];
            h1[0]=hole_[ch][0]; h1[1]=hole_[ch][1];
            h2[0]=hole_[bst][0]; h2[1]=hole_[bst][1];
            for (int j=0;j<bcnt_;j++) { h1[2+j]=board_[j]; h2[2+j]=board_[j]; }
            int cmp = eval7(h1).compare(eval7(h2));
            if (cmp>0) { winners_.clear(); winners_.push_back(ch); }
            else if (cmp==0) winners_.push_back(ch);
        }
        if (!winners_.empty()) {
            int share=pot_/(int)winners_.size(), rem=pot_%(int)winners_.size();
            for (int w : winners_) stacks_[w]+=share;
            if (rem>0) stacks_[winners_[0]]+=rem;
            pot_=0;
        }
    }

public:
    GameEngine(int sb, int bb, int stack, int nplayers=6)
        : sb_(sb), bb_(bb), init_stack_(stack), np_(nplayers),
          bcnt_(0), pot_(0), cbet_(0), dpos_(0), cplayer_(0), round_(0), acnt_(0) {
        std::memset(stacks_,0,sizeof(stacks_));
        std::memset(bets_,0,sizeof(bets_));
        std::memset(active_,0,sizeof(active_));
        std::memset(allin_,0,sizeof(allin_));
        std::memset(hole_,-1,sizeof(hole_));
        std::memset(hcnt_,0,sizeof(hcnt_));
        std::memset(board_,-1,sizeof(board_));
        for (int i=0;i<np_;i++) stacks_[i]=init_stack_;
    }

    // ---- game control ----
    // NOTE: start_new_hand / handle_action / next_player 等纯 C++ 函数
    // 通过 pybind11 call_guard<py::gil_scoped_release> 释放 GIL
    void start_new_hand(int dealer_pos) {
        dpos_ = dealer_pos % np_;
        deck_.shuffle();
        bcnt_=0; std::memset(board_,-1,sizeof(board_));
        pot_=0; cbet_=0; acnt_=0; winners_.clear();
        std::memset(bets_,0,sizeof(bets_));
        for (int i=0;i<np_;i++) {
            active_[i]=true; allin_[i]=false;
            stacks_[i]=init_stack_;
            hole_[i][0]=hole_[i][1]=-1; hcnt_[i]=0;
        }
        post_blinds(); deal_hole(); round_=1; setup_round();
    }

    bool handle_action(int pi, int action, int amount) {
        if (pi<0||pi>=np_||!active_[pi]||allin_[pi]) return false;
        if (action==0) { active_[pi]=false; acnt_++; return true; }
        if (action==1) { // call
            int ca=std::max(0, cbet_-bets_[pi]);
            if (ca>stacks_[pi]) { // all-in
                int a=stacks_[pi]; bets_[pi]+=a; stacks_[pi]=0; allin_[pi]=true; pot_+=a;
            } else {
                bets_[pi]+=ca; stacks_[pi]-=ca; pot_+=ca;
                if (stacks_[pi]==0) allin_[pi]=true;
            }
            acnt_++; return true;
        }
        if (action==2) { // raise
            if (amount<=cbet_) return false;
            int mr=std::max(cbet_*2, bb_);
            if (amount<mr && amount<stacks_[pi]) return false;
            int ra=amount-bets_[pi];
            if (ra>stacks_[pi]) return false;
            bets_[pi]+=ra; stacks_[pi]-=ra; pot_+=ra; cbet_=amount;
            if (stacks_[pi]==0) allin_[pi]=true;
            acnt_=1; return true;
        }
        return false;
    }

    int next_player() {
        for (int i=0;i<np_;i++) {
            cplayer_=(cplayer_+1)%np_;
            if (active_[cplayer_]&&!allin_[cplayer_]) return cplayer_;
        }
        return -1;
    }

    bool is_round_over() const {
        int n=0;
        for (int i=0;i<np_;i++) if (active_[i]&&!allin_[i]) n++;
        if (n==0) return true;
        if (acnt_>=n) {
            for (int i=0;i<np_;i++)
                if (active_[i]&&!allin_[i]&&bets_[i]!=cbet_) return false;
            return true;
        }
        return false;
    }

    bool is_hand_over() const {
        int n=0; for (int i=0;i<np_;i++) if (active_[i]) n++;
        return n<=1 || round_>4;
    }

    bool proceed_to_next_round() {
        int ai=-1, ac=0;
        for (int i=0;i<np_;i++) if (active_[i]) { ac++; ai=i; }
        if (ac==1) { award_pot_to(ai); return true; }
        if (round_==1) { for(int i=0;i<3;i++) board_[bcnt_++]=deck_.deal(); round_=2; setup_round(); }
        else if (round_==2) { board_[bcnt_++]=deck_.deal(); round_=3; setup_round(); }
        else if (round_==3) { board_[bcnt_++]=deck_.deal(); round_=4; setup_round(); }
        else if (round_==4) { do_showdown(); return true; }
        return false;
    }

    void award_pot_to(int pi) {
        stacks_[pi]+=pot_; pot_=0;
        winners_.clear(); winners_.push_back(pi);
    }

    // ---- state vector (134 dims) ----
    py::array_t<float> get_state_vector(int pi) const {
        auto out = py::array_t<float>(134);
        float* v = out.mutable_data();
        std::memset(v, 0, 134*sizeof(float));

        // [0:52] hand one-hot
        for (int i=0;i<hcnt_[pi];i++) { int c=hole_[pi][i]; if (c>=0&&c<52) v[c]=1.f; }
        // [52:104] board one-hot
        for (int i=0;i<bcnt_;i++) { int c=board_[i]; if (c>=0&&c<52) v[52+c]=1.f; }
        // [104:122] player info (hero first, rotated)
        float bb = std::max((float)bb_, 1.f);
        for (int i=0;i<np_;i++) {
            int t=(pi+i)%np_, base=104+i*3;
            v[base]   = std::min((float)stacks_[t]/(bb*100.f), 1.f);
            v[base+1] = std::min((float)bets_[t]/(bb*100.f), 1.f);
            v[base+2] = active_[t] ? 1.f : 0.f;
        }
        // [122:127] game info
        v[122] = (round_-1)/4.f;
        v[123] = std::min((float)pot_/(bb*100.f), 1.f);
        v[124] = std::min((float)cbet_/(bb*100.f), 1.f);
        v[125] = bb_/100.f;
        v[126] = sb_/100.f;
        // [127:133] position one-hot
        int dist = (pi-dpos_+np_)%np_;
        if (dist<6) v[127+dist]=1.f;
        // [133] active count normalized
        int ac=0; for (int i=0;i<np_;i++) if (active_[i]) ac++;
        v[133] = ac>0 ? (ac-1)/5.f : 0.f;
        return out;
    }

    // ---- legal mask (5 dims) ----
    py::array_t<float> get_legal_mask(int pi) const {
        auto out = py::array_t<float>(5);
        float* m = out.mutable_data();
        std::memset(m, 0, 5*sizeof(float));
        if (!active_[pi]||allin_[pi]) { m[0]=1.f; return out; }
        int ca = std::max(0, cbet_-bets_[pi]);
        m[0]=1.f;
        if (ca<=stacks_[pi]) m[1]=1.f;
        int cb=std::max(cbet_,bb_), r2=cb*2, r4=cb*4;
        int ai=bets_[pi]+stacks_[pi], mr=std::max(cbet_*2,bb_);
        if (r2>cbet_&&r2>=mr&&(r2-bets_[pi])<=stacks_[pi]) m[2]=1.f;
        if (r4>cbet_&&r4>=mr&&(r4-bets_[pi])<=stacks_[pi]) m[3]=1.f;
        if (ai>cbet_&&ai>=mr&&stacks_[pi]>0) m[4]=1.f;
        float s=0; for(int i=0;i<5;i++) s+=m[i];
        if (s<=0) { m[0]=1.f; if (ca<=stacks_[pi]) m[1]=1.f; }
        return out;
    }

    // ---- hand strength (pure C++, GIL released via call_guard) ----
    float get_hand_strength(int pi) const {
        if (hcnt_[pi]<2) return 0.f;
        if (bcnt_==0) {
            // preflop formula
            int r1=rank_val(hole_[pi][0]), r2=rank_val(hole_[pi][1]);
            int s1=suit_of(hole_[pi][0]), s2=suit_of(hole_[pi][1]);
            if (r1<r2) std::swap(r1,r2);
            bool suited=(s1==s2), pair=(r1==r2);
            auto rs=[](int v)->float{
                if(v==14)return 10.f; if(v==13)return 8.f; if(v==12)return 7.f;
                if(v==11)return 6.f; if(v==10)return 5.f; return v/2.f;};
            float sc=rs(r1);
            if (pair) { sc*=2; sc=std::max(5.f,sc); }
            if (suited) sc+=2.f;
            if (!pair) {
                int gap=r1-r2-1;
                if(gap==1)sc-=1.f; else if(gap==2)sc-=2.f;
                else if(gap==3)sc-=4.f; else if(gap>=4)sc-=5.f;
                if(gap<=1&&r1<12) sc+=1.f;
            }
            return std::min(1.f, sc/20.f);
        }
        // postflop MC (5 sims)
        bool known[52]={}; 
        known[hole_[pi][0]]=true; known[hole_[pi][1]]=true;
        for (int i=0;i<bcnt_;i++) known[board_[i]]=true;
        int8_t rem[52]; int nr=0;
        for (int i=0;i<52;i++) if (!known[i]) rem[nr++]=(int8_t)i;
        int nopp=0; for (int i=0;i<np_;i++) if (i!=pi&&active_[i]) nopp++;
        if (nopp==0) nopp=1;
        thread_local std::mt19937 rng(std::random_device{}());
        int wins=0, needed=5-bcnt_;
        for (int sim=0;sim<5;sim++) {
            std::shuffle(rem, rem+nr, rng);
            int idx=0;
            int8_t sb[5];
            for (int i=0;i<bcnt_;i++) sb[i]=board_[i];
            for (int i=0;i<needed;i++) sb[bcnt_+i]=rem[idx++];
            int8_t hh[7]; hh[0]=hole_[pi][0]; hh[1]=hole_[pi][1];
            for (int i=0;i<5;i++) hh[2+i]=sb[i];
            HandRank hr=eval7(hh);
            bool won=true;
            for (int o=0;o<nopp&&won;o++) {
                int8_t oh[7]; oh[0]=rem[idx++]; oh[1]=rem[idx++];
                for (int i=0;i<5;i++) oh[2+i]=sb[i];
                if (eval7(oh).compare(hr)>=0) won=false;
            }
            if (won) wins++;
        }
        return (float)wins/5.f;
    }

    // ---- property accessors ----
    int  get_pot()       const { return pot_; }
    int  get_cbet()      const { return cbet_; }
    int  get_round()     const { return round_; }
    int  get_dpos()      const { return dpos_; }
    int  get_np()        const { return np_; }
    int  get_bb()        const { return bb_; }
    int  get_sb()        const { return sb_; }
    int  get_init_stack() const { return init_stack_; }
    int  player_stack(int i)     const { return stacks_[i]; }
    int  player_bet(int i)       const { return bets_[i]; }
    bool player_active(int i)    const { return active_[i]; }
    bool player_allin(int i)     const { return allin_[i]; }

    py::array_t<int8_t> player_card_codes(int i) const {
        int n=hcnt_[i]; auto out=py::array_t<int8_t>(n);
        int8_t* d=out.mutable_data();
        for (int j=0;j<n;j++) d[j]=hole_[i][j];
        return out;
    }
    py::array_t<int8_t> board_codes() const {
        auto out=py::array_t<int8_t>(bcnt_);
        int8_t* d=out.mutable_data();
        for (int j=0;j<bcnt_;j++) d[j]=board_[j];
        return out;
    }
    py::list winner_indices() const {
        py::list r; for (int w : winners_) r.append(w); return r;
    }
};

// ================================================================
//  Standalone decode_action (matches Python StateRepresentation.decode_action)
// ================================================================
static py::tuple decode_action_fn(int aidx, int cbet, int bb, int pbet, int pstack) {
    if (aidx==0) return py::make_tuple("fold", 0);
    if (aidx==1) { int ca=std::max(0,cbet-pbet); return py::make_tuple("call",ca); }
    int cb=std::max(cbet,bb);
    if (aidx==2) { int a=std::min(cb*2, pbet+pstack); return py::make_tuple("raise",a); }
    if (aidx==3) { int a=std::min(cb*4, pbet+pstack); return py::make_tuple("raise",a); }
    if (aidx==4) return py::make_tuple("raise", pbet+pstack);
    return py::make_tuple("fold", 0);
}

// ================================================================
//  pybind11 module
// ================================================================
PYBIND11_MODULE(poker_engine_cc, m) {
    m.doc() = "High-performance C++ poker engine for SEED RL training";

    py::class_<GameEngine>(m, "GameEngine")
        .def(py::init<int,int,int,int>(),
             py::arg("small_blind"), py::arg("big_blind"),
             py::arg("initial_stack"), py::arg("num_players")=6)
        // 纯 C++ 计算函数：释放 GIL，让其他 Worker 可以并行执行
        .def("start_new_hand",      &GameEngine::start_new_hand,
             py::call_guard<py::gil_scoped_release>())
        .def("handle_action",       &GameEngine::handle_action,
             py::call_guard<py::gil_scoped_release>())
        .def("next_player",         &GameEngine::next_player,
             py::call_guard<py::gil_scoped_release>())
        .def("is_round_over",       &GameEngine::is_round_over,
             py::call_guard<py::gil_scoped_release>())
        .def("is_hand_over",        &GameEngine::is_hand_over,
             py::call_guard<py::gil_scoped_release>())
        .def("proceed_to_next_round",&GameEngine::proceed_to_next_round,
             py::call_guard<py::gil_scoped_release>())
        .def("award_pot_to",        &GameEngine::award_pot_to,
             py::call_guard<py::gil_scoped_release>())
        // 以下函数返回 numpy/Python 对象，需要持有 GIL —— 不加 call_guard
        .def("get_state_vector",    &GameEngine::get_state_vector)
        .def("get_legal_mask",      &GameEngine::get_legal_mask)
        // hand_strength 返回标量 float，可安全释放 GIL
        .def("get_hand_strength",   &GameEngine::get_hand_strength,
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("pot",          &GameEngine::get_pot)
        .def_property_readonly("current_bet",  &GameEngine::get_cbet)
        .def_property_readonly("round",        &GameEngine::get_round)
        .def_property_readonly("dealer_pos",   &GameEngine::get_dpos)
        .def_property_readonly("num_players",  &GameEngine::get_np)
        .def_property_readonly("big_blind",    &GameEngine::get_bb)
        .def_property_readonly("small_blind",  &GameEngine::get_sb)
        .def_property_readonly("initial_stack",&GameEngine::get_init_stack)
        .def("get_player_stack",     &GameEngine::player_stack)
        .def("get_player_bet",       &GameEngine::player_bet)
        .def("get_player_is_active", &GameEngine::player_active)
        .def("get_player_is_all_in", &GameEngine::player_allin)
        .def("get_player_card_codes",&GameEngine::player_card_codes)
        .def("get_board_codes",      &GameEngine::board_codes)
        .def("get_winner_indices",   &GameEngine::winner_indices);

    m.def("decode_action", &decode_action_fn,
          "Decode action index → (action_str, amount)",
          py::arg("action_idx"), py::arg("current_bet"), py::arg("big_blind"),
          py::arg("player_bet"), py::arg("player_stack"));
}
