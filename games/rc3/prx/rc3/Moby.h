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
    Vec4 coll_pos;
    Vec4 position;
    char state;
    char group;
    char mClass;
    char alpha;
    MobyClass *pClass;
    void *pChain;
    float scale;
    char update_distance;
    char enabled;
    short draw_distance;
    unsigned short mode_bits;
    unsigned short field19_0x36;
    unsigned int stateTimerMaybe;
    char field21_0x3c;
    char field22_0x3d;
    char field23_0x3e;
    char field24_0x3f;
    Vec4 rotation;
    char field26_0x50;
    char animationFrame;
    u8 updateID;
    u8 animationID;
    float field30_0x54;
    float field34_0x58;
    float field35_0x5c;
    void *field36_0x60;
    void *update_ptr;
    void *vars;
    void *field42_0x6c;
    char field43_0x70;
    char field44_0x71;
    char field45_0x72;
    char field46_0x73;
    void *field47_0x74;
    void *field48_0x78;
    char field49_0x7c;
    char field50_0x7d;
    char field51_0x7e;
    char animStateMaybe;
    unsigned int field53_0x80;
    int field54_0x84;
    int field55_0x88;
    char field56_0x8c;
    char field57_0x8d;
    char field58_0x8e;
    char field59_0x8f;
    struct Moby *parent;
    unsigned short *collision;
    float *collisionMesh;
    unsigned int field63_0x9c;
    char field64_0xa0;
    char field65_0xa1;
    char field66_0xa2;
    char field67_0xa3;
    char field68_0xa4;
    char field69_0xa5;
    int field70_0xa6;
    short o_class;
    unsigned int field72_0xac;
    char field73_0xb0;
    char field74_0xb1;
    unsigned short UID;
    char field76_0xb4;
    char field77_0xb5;
    char field78_0xb6;
    char field79_0xb7;
    struct Moby *field80_0xb8;
    char field81_0xbc;
    char field82_0xbd;
    char field83_0xbe;
    char field84_0xbf;
    Vec4 forward;
    Vec4 left;
    Vec4 up;
    Vec4 something;

#ifdef __cplusplus

#endif
};

#endif //RAC3_GYM_MOBY_H
