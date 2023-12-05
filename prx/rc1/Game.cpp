//
// Created by Vetle Hjelle on 20/12/2022.
//

#include "Game.h"

#include <rc1/rc1.h>

#include <lib/vector.h>
#include <lib/logger.h>
#include <lib/memory.h>

#include <sys/random_number.h>
#include <sysutil/sysutil_msgdialog.h>

#include "views/RemoteView.h"

#include "Player.h"
#include "PersistentStorage.h"
#include "GoldBolt.h"

float sqrt_ppc(float number) {
    float result;
    asm("fsqrts %0, %1" : "=f"(result) : "f"(number));
    return result;
}

float distance(const Vec4& a, const Vec4& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrt_ppc(dx * dx + dy * dy + dz * dz);
}

// For whatever dumb reason I can't get the compiler to include
//  .cpp files under /lib/, so it's defined here.
// Maybe better to just have it here anyway. Idk.
LogLevel Logger::log_level_ = Debug;

#define custom_frame_count *((int*)0xB00000)
#define progress_frame_count *((int*)0xB00004)

#define hoverboard_lady_address *((uint32_t*)0xB00020)
#define skid_address *((uint32_t*)0xB00024)

#define collision_ahead *((float*)0xB00030)
#define collision_up *((float*)0xB00034)
#define collision_left *((float*)0xB00038)
#define collision_right *((float*)0xB0003c)
#define collision_down *((float*)0xB00050)

#define collision_class_ahead *((int*)0xB00040)
#define collision_class_up *((int*)0xB00044)
#define collision_class_left *((int*)0xB00048)
#define collision_class_right *((int*)0xB0004c)
#define collision_class_down *((int*)0xB00054)

struct Vector2D {
    double x;
    double y;

    Vector2D(double x_val, double y_val) : x(x_val), y(y_val) {}
};

//Vec3 Normalize(Vec3 &v) {
//    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
//
//    Vec3 normalized = Vec3();
//    normalized.x = v.x / len;
//    normalized.y = v.y / len;
//    normalized.z = v.z / len;
//
//    return normalized;
//}

//// Function to calculate the vector ahead of a given vector
//Vector2D calculateVectorAhead(const Vector2D& inputVector, double rotationRadians, double distance) {
//    // Calculate the new x and y components
//    double newX = inputVector.x + distance * std::cos(rotationRadians);
//    double newY = inputVector.y - distance * std::sin(rotationRadians);
//
//    return Vector2D(newX, newY);
//}

void Game::start() {
    hoverboard_lady_address = 0;
    skid_address = 0;
}

void Game::on_tick() {
    // Pass inputs to the current view and send to server
    if (pressed_buttons) {
        if (current_view) {
            Logger::trace("Pressed buttons (%08X) sent to view", pressed_buttons);
            current_view->on_pressed_buttons(pressed_buttons);
        }
    }

//    if (current_planet == 0 && game_state != PlayerControl) {
//        game_state = PlayerControl;
//    }

    // Player is most likely on intro menu scene. Present and handle multiplayer startup stuff.
    if (current_planet == 0 && frame_count > 10) {
        seen_planets[0] = 1;
        seen_planets[5] = 1;
        *(int*)0xa10700 = 1;
        *(int*)0xa10704 = (int)5;
        //*(int*)0x969c70 = (int)5;
    }

    if (hoverboard_lady_address == 0) {
        hoverboard_lady_address = (uint32_t)Moby::find_first(918);
    } else {
        Moby* hoverboard_lady = (Moby*)hoverboard_lady_address;
        if (hoverboard_lady->oClass != 918 || hoverboard_lady->state >= 0x7f) {
            hoverboard_lady_address = (uint32_t)Moby::find_first(918);
        }
    }

    if (skid_address == 0) {
        skid_address = (uint32_t)Moby::find_first(717);
    } else {
        Moby* skid = (Moby*)skid_address;
        if (skid->oClass != 717 || skid->state >= 0x7f) {
            skid_address = (uint32_t)Moby::find_first(717);
        }
    }

    if (!current_view) {
        RemoteView* view = new RemoteView();
        this->transition_to(view);
    } else if (current_planet != 0 && frame_count > 60) {
        RemoteView* view = (RemoteView*)current_view;

////        Vec3 rot = Normalize(*(Vec3*)&player_rot);
//
//        Vec3 forward = Vec3();
////        forward.z = std::cos(rot.y) * std::sin(rot.z);
////        forward.x = -std::sin(rot.z);
////        forward.y = std::cos(rot.y) * std::cos(rot.z);
//        forward.x = cos_approx(player_rot.z);
//        forward.y = sin_approx(player_rot.z);
//        forward.z = 0;
//
//        Vec4 ahead = Vec4();
//        ahead.x = player_pos.x + 20.0f * forward.x;
//        ahead.y = player_pos.y + 20.0f * forward.y;
//        ahead.z = player_pos.z + 0.5f;
//        ahead.w = 1;
//
////        if (test_moby == nullptr) {
////            test_moby = Moby::spawn(500, 0, 0);
////            test_moby->pUpdate = nullptr;
////        }
////
////        test_moby->position.x = ahead.x;
////        test_moby->position.y = ahead.y;
////        test_moby->position.z = player_pos.z;
//
////        Logger::info("Foward: x: %.2f y: %.2f z: %.2f. Position: x1: %.2f y1: %.2f z1: %.2f; Forward: x2: %.2f y2: %.2f z2: %.2f",
////                     forward.x, forward.y, forward.z,
////                     player_pos.x, player_pos.y, player_pos.z,
////                     ahead.x, ahead.y, ahead.z);
//
//        int coll = coll_line(&player_pos, &ahead, 0x4, ratchet_moby);

//        Vec3 forward = Vec3();
//        forward.x = cos_approx(player_rot.y) * cos_approx(player_rot.z);
//        forward.y = cos_approx(player_rot.y) * sin_approx(player_rot.z);
//        forward.z = -sin_approx(player_rot.y);
//
//        if (game_state == PlayerControl) {
//            Logger::debug("%f, %f, %f", forward.x, forward.y, forward.z);
//        }

        Vec4 forward = ratchet_moby->forward;

        // Calculate left and right directions
        Vec4 right = ratchet_moby->right;

        Vec4 left = Vec4();
        left.x = -right.x;
        left.y = -right.y;
        left.z = -right.z;
        left.w = right.w;

        float ray_distance = 64.0f;
        float ray_wide = 64.0f;

        float fov = 64.0f;
        int rows = 64;
        int cols = 64;

        // Raycast in a grid pattern extending from the player with the given fov
        for (int i = 0; i <= rows; i++) {
            for (int j = 0; j <= cols; j++) {
                Vec4 ray = Vec4();
                ray.x = player_pos.x + ray_distance * forward.x + (fov/rows) * i * left.x + (fov/cols) * j * right.x;
                ray.y = player_pos.y + ray_distance * forward.y + (fov/rows) * i * left.y + (fov/cols) * j * right.y;
                ray.z = player_pos.z + ray_distance * forward.z + (fov/rows) * i * left.z + (fov/cols) * j * right.z;
                ray.w = 1;


            }
        }

        Vec4 ahead = Vec4();
        ahead.x = player_pos.x + ray_distance * forward.x;
        ahead.y = player_pos.y + ray_distance * forward.y;
        ahead.z = player_pos.z + ray_distance * forward.z;
        ahead.w = 1;

        Vec4 ahead_up = Vec4();
        ahead_up.x = player_pos.x + ray_distance * forward.x + (ray_wide/2) * ratchet_moby->up.x;
        ahead_up.y = player_pos.y + ray_distance * forward.y + (ray_wide/2) * ratchet_moby->up.y;
        ahead_up.z = player_pos.z + ray_distance * forward.z + (ray_wide/2) * ratchet_moby->up.z;
        ahead_up.w = 1;

        // Calculate ahead_left and ahead_right
        Vec4 ahead_left = Vec4();
        ahead_left.x = player_pos.x + ray_distance * forward.x + ray_wide * left.x;
        ahead_left.y = player_pos.y + ray_distance * forward.y + ray_wide * left.y;
        ahead_left.z = player_pos.z + ray_distance * forward.z + ray_wide * left.z;
        ahead_left.w = 1;

        Vec4 ahead_right = Vec4();
        ahead_right.x = player_pos.x + ray_distance * forward.x + ray_wide * right.x;
        ahead_right.y = player_pos.y + ray_distance * forward.y + ray_wide * right.y;
        ahead_right.z = player_pos.z + ray_distance * forward.z + ray_wide * right.z;
        ahead_right.w = 1;

        Vec4 ahead_down = Vec4();
        ahead_down.x = player_pos.x + ray_distance * forward.x - (ray_wide/2) * ratchet_moby->up.x;
        ahead_down.y = player_pos.y + ray_distance * forward.y - (ray_wide/2) * ratchet_moby->up.y;
        ahead_down.z = player_pos.z + ray_distance * forward.z - (ray_wide/2) * ratchet_moby->up.z;
        ahead_down.w = 1;

//        if (test_moby == nullptr) {
//            test_moby = Moby::spawn(500, 0, 0);
//            test_moby->pUpdate = nullptr;
//            test_moby->collision = nullptr;
//        }
//
//        test_moby->position.x = ahead_down.x;
//        test_moby->position.y = ahead_down.y;
//        test_moby->position.z = ahead_down.z;
//
//        if (test_moby_a == nullptr) {
//            test_moby_a = Moby::spawn(500, 0, 0);
//            test_moby_a->pUpdate = nullptr;
//            test_moby_a->collision = nullptr;
//        }
//
//        test_moby_a->position.x = ahead_up.x;
//        test_moby_a->position.y = ahead_up.y;
//        test_moby_a->position.z = ahead_up.z;
//
//        if (test_moby_l == nullptr) {
//            test_moby_l = Moby::spawn(500, 0, 0);
//            test_moby_l->pUpdate = nullptr;
//            test_moby_l->collision = nullptr;
//        }
//
//        test_moby_l->position.x = ahead_left.x;
//        test_moby_l->position.y = ahead_left.y;
//        test_moby_l->position.z = ahead_left.z;
//
//        if (test_moby_r == nullptr) {
//            test_moby_r = Moby::spawn(500, 0, 0);
//            test_moby_r->pUpdate = nullptr;
//            test_moby_r->collision = nullptr;
//        }
//
//        test_moby_r->position.x = ahead_right.x;
//        test_moby_r->position.y = ahead_right.y;
//        test_moby_r->position.z = ahead_right.z;

        collision_ahead = -32.0f;
        collision_up = -32.0f;
        collision_down = -32.0f;
        collision_left = -32.0f;
        collision_right = -32.0f;

        collision_class_ahead = 0;
        collision_class_up = 0;
        collision_class_down = 0;
        collision_class_left = 0;
        collision_class_right = 0;

        Vec4 source_vect = Vec4(player_pos.x, player_pos.y, player_pos.z + 0.5f, player_pos.w);

        // Collision checks
        int coll_forward = coll_line(&source_vect, &ahead, 0x24, ratchet_moby, nullptr);

        view->coll_class = 0;

        if (coll_forward) {
            collision_ahead = distance(source_vect, coll_output.ip);

            if (coll_output.pMoby) {
                collision_class_ahead = coll_output.pMoby->oClass;
                view->coll_class = coll_output.pMoby->oClass;
            }
        }

        int coll_up = coll_line(&source_vect, &ahead_up, 0x24, ratchet_moby, nullptr);

        view->coll_up_class = 0;

        if (coll_up) {
            collision_up = distance(source_vect, coll_output.ip);

            if (coll_output.pMoby) {
            collision_class_up = coll_output.pMoby->oClass;
            view->coll_up_class = coll_output.pMoby->oClass;
            }
        }

        int coll_down = coll_line(&source_vect, &ahead_down, 0x24, ratchet_moby, nullptr);

        view->coll_down_class = 0;

        if (coll_down) {
            collision_down = distance(source_vect, coll_output.ip);

            if (coll_output.pMoby) {
                collision_class_down = coll_output.pMoby->oClass;
                view->coll_down_class = coll_output.pMoby->oClass;
            }
        }

        int coll_left = coll_line(&source_vect, &ahead_left, 0x24, ratchet_moby, nullptr);

        view->coll_left_class = 0;

        if (coll_left) {
            collision_left = distance(source_vect, coll_output.ip);

            if (coll_output.pMoby) {
                collision_class_left = coll_output.pMoby->oClass;
                view->coll_left_class = coll_output.pMoby->oClass;
            }
        }

        int coll_right = coll_line(&source_vect, &ahead_right, 0x24, ratchet_moby, nullptr);

        view->coll_right_class = 0;

        if (coll_right) {
            collision_right = distance(source_vect, coll_output.ip);

            if (coll_output.pMoby) {
                collision_class_right = coll_output.pMoby->oClass;
                view->coll_right_class = coll_output.pMoby->oClass;
            }
        }

        view->coll = collision_ahead;
        view->coll_up = collision_up;
        view->coll_down = collision_down;
        view->coll_left = collision_left;
        view->coll_right = collision_right;
    }

    custom_frame_count += 1;

    while (progress_frame_count != custom_frame_count && current_planet != 0) {

    }

    return;
}

void Game::before_player_spawn() {

}

void Game::on_render() {
    // If loading, we shouldn't render anything;
    if (game_state == 6) {
        return;
    }

    if (current_view) {
        current_view->render();
    }

    return;
}

void Game::transition_to(View *view) {
    Logger::trace("Starting transition to a new view");

    if (current_view) {
        current_view->on_unload();
        delete current_view;
    }

    current_view = view;
    current_view->on_load();

    Logger::trace("Done transitioning");
}

void Game::alert(String& message) {
    cellMsgDialogOpen2(CELL_MSGDIALOG_TYPE_SE_TYPE_NORMAL, message.c_str(), nullptr, nullptr, nullptr);
}


extern "C" void _c_game_tick() {
    Game::shared().on_tick();
}

extern "C" void _c_game_render() {
    Game::shared().on_render();
}

extern "C" void _c_game_start() {
    Game::shared().start();
}

extern "C" void _c_game_quit() {

}

extern "C" void _c_on_respawn() {
    Game::shared().before_player_spawn();
}