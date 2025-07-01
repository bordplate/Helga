//
// Created by Vetle Hjelle on 19/12/2022.
//

#ifndef PS3_CKIT_BRIDGING_H
#define PS3_CKIT_BRIDGING_H

#include "Moby.h"

#ifdef __cplusplus
extern "C" {
#endif

    void _c_on_game_start();
    void _c_game_tick();
    void _c_on_flag_update(struct Moby* flag_moby);


#ifdef __cplusplus
}
#endif

#endif //PS3_CKIT_BRIDGING_H
