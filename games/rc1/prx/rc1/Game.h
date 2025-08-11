//
// Created by Vetle Hjelle on 20/12/2022.
//

#ifndef PS3_CKIT_GAME_H
#define PS3_CKIT_GAME_H

#include <rc1/common.h>

#include "View.h"

#define death_count *((int*)0xB00500)

struct Game {
public:
    static Game& shared() {
        static Game game;
        return game;
    }

    int userid;

    void start();

    void transition_to(View* view);

    void on_tick();
    void on_render();

    void before_player_spawn();

    void alert(String& message);
private:
    Game() {
    }
    Game(Game const&);

    View* current_view;

//    Moby* test_moby;
//    Moby* test_moby_a;
//    Moby* test_moby_r;
//    Moby* test_moby_l;

    int last_death_count;

    Moby* camera_moby;

    Moby* checkpoint_moby;
    float checkpoint_bounce_z;

	unsigned short *checkpoint_collision;

    bool oscillation_direction_x;
    bool oscillation_direction_y;
};


#endif //PS3_CKIT_GAME_H
