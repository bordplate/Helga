//
// Created by Vetle Hjelle on 17/12/2023.
//

#ifndef RAC3_GYM_MOBY_H
#define RAC3_GYM_MOBY_H

#include <lib/shk.h>
#include <lib/types.h>

typedef struct {
    u16 uuid;
    char next_animation_id;
    char o_class;
    int animation_duration;
    u16 sig;
    u8 collision_debounce;
} MPMobyVars;

typedef struct MobySeq { /* From Deadlocked types */
    undefined field0_0x0;
    undefined field1_0x1;
    undefined field2_0x2;
    undefined field3_0x3;
    undefined field4_0x4;
    undefined field5_0x5;
    undefined field6_0x6;
    undefined field7_0x7;
    undefined field8_0x8;
    undefined field9_0x9;
    undefined field10_0xa;
    undefined field11_0xb;
    undefined field12_0xc;
    undefined field13_0xd;
    undefined field14_0xe;
    undefined field15_0xf;
    u8 frame_count;
    u8 next_seq;
    u8 trigsCnt;
    u8 pad;
    short *trigs;
    void *animInfo;
    void *frameData;
} MobySeq;

typedef struct MobyClass { /* MobyClass from Deadlocked Types, probably very wrong */
    void *packets;
    u8 pakcet_cnt_0;
    u8 packet_cnt_1;
    u8 metal_cnt;
    u8 metal_ofs;
    u8 joint_cnt;
    u8 pad;
    u8 packet_cnt_2;
    u8 team_texs; /* Obvs wrong, ain't no teams in Rac1 */
    u8 seq_cnt;
    u8 sound_cnt;
    u8 lod_trans;
    u8 shadow;
    u16 *collision;
    void *skeleton;
    void *common_trans;
    void *anim_joints;
    void *gif_usage;
    float gScale;
    void *SoundDefs;
    char bangles;
    char mip_dist;
    short corncob;
    undefined field23_0x30;
    undefined field24_0x31;
    undefined field25_0x32;
    undefined field26_0x33;
    undefined field27_0x34;
    undefined field28_0x35;
    undefined field29_0x36;
    undefined field30_0x37;
    undefined field31_0x38;
    undefined field32_0x39;
    undefined field33_0x3a;
    undefined field34_0x3b;
    undefined field35_0x3c;
    undefined field36_0x3d;
    undefined field37_0x3e;
    undefined field38_0x3f;
    struct Moby *unk_ptr;
    u32 mode_bits;
    char type;
    char mode_bits2;
    struct MobySeq *seqs;
} MobyClass;

#ifdef __cplusplus
#pragma pack(push, 1)
#endif

struct Moby {
    struct Vec4 bSphere;
    struct Vec4 pos;
    char state;
    char group;
    char mClass;
    char alpha;
    void *MobyClass;
    void *pChain;
    float scale;
    char updateDist;
    char drawn;
    u16 modeBits;
    u16 modeBits2;
    uint64_t lights;
    void *MobySeq;
    char field15_0x42;
    char *field16_0x43;
    char field17_0x47;
    char field18_0x48;
    char field19_0x49;
    short animIScale;
    short poseCacheEntryIndex;
    short animFlags;
    char lSeq;
    char jointCnt;
    void *jointCache;
    void *pManipulator;
    int glow_rgba;
    char lod_trans;
    char lod_trans2;
    char metal;
    char subState;
    char prevState;
    char stateType;
    void *pUpdate;
    void *pVars;
    char field36_0x6c;
    char field37_0x6d;
    char field38_0x6e;
    char field39_0x6f;
    struct Vec4 field40_0x70;
    char field41_0x80;
    char field42_0x81;
    char field43_0x82;
    char field44_0x83;
    char field45_0x84;
    char field46_0x85;
    char field47_0x86;
    char field48_0x87;
    char field49_0x88;
    char field50_0x89;
    char field51_0x8a;
    char field52_0x8b;
    char field53_0x8c;
    char field54_0x8d;
    char field55_0x8e;
    char field56_0x8f;
    char field57_0x90;
    char field58_0x91;
    char field59_0x92;
    char field60_0x93;
    char field61_0x94;
    char field62_0x95;
    char field63_0x96;
    char field64_0x97;
    int field65_0x98;
    char field66_0x9c;
    char field67_0x9d;
    char field68_0x9e;
    char field69_0x9f;
    int field70_0xa0;
    char field71_0xa4;
    char field72_0xa5;
    char field73_0xa6;
    char field74_0xa7;
    char field75_0xa8;
    char field76_0xa9;
    u16 o_class;
    char field78_0xac;
    char field79_0xad;
    char field80_0xae;
    char field81_0xaf;
    char field82_0xb0;
    char field83_0xb1;
    short UID;
    char field85_0xb4;
    char field86_0xb5;
    char field87_0xb6;
    char field88_0xb7;
    char field89_0xb8;
    char field90_0xb9;
    char field91_0xba;
    char field92_0xbb;
    char field93_0xbc;
    char field94_0xbd;
    char field95_0xbe;
    char field96_0xbf;
    struct Vec4 forward;
    struct Vec4 right;
    struct Vec4 up;
    struct Vec4 rot;
};

#endif //RAC3_GYM_MOBY_H
