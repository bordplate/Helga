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
#include <sys/ppu_thread.h>

#include "views/RemoteView.h"

#include "Player.h"
#include "PersistentStorage.h"
#include "GoldBolt.h"

float sqrt_ppc(float number) {
    float result;
    asm("fsqrts %0, %1" : "=f"(result) : "f"(number));
    return result;
}

double sqrt_approx(double number, double tolerance = 1e-10) {
    if (number < 0) {
        return -1; // Return -1 for negative numbers as error indicator
    }
    if (number == 0) {
        return 0; // The square root of 0 is 0
    }

    double estimate = number;
    double diff = tolerance;

    while (diff >= tolerance) {
        double new_estimate = 0.5 * (estimate + number / estimate);
        diff = (estimate > new_estimate) ? estimate - new_estimate : new_estimate - estimate;
        estimate = new_estimate;
    }

    return estimate;
}

float distance(const Vec4& a, const Vec4& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrt_approx(dx * dx + dy * dy + dz * dz);
//    return sqrt_ppc(dx * dx + dy * dy + dz * dz);
}

float sin_approx(float x) {
    return x - x*x*x/6 + x*x*x*x*x/120;
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

#define oscillation_offset_x *((float*)0xB00060)
#define oscillation_offset_y *((float*)0xB00064)

#define collisions_mobys ((Moby**)0xB00300)

#define checkpoint_position ((Vec4*)0xB00400)

#define BASE_ADDRESS 0xB00600

#define collisions_distance  ((float*)(BASE_ADDRESS + (0x100 * 4 * 0)))
#define collisions_class     ((int*)(BASE_ADDRESS + (0x100 * 4 * 1)))
#define collisions_normals_x ((float*)(BASE_ADDRESS + (0x100 * 4 * 2)))
#define collisions_normals_y ((float*)(BASE_ADDRESS + (0x100 * 4 * 3)))
#define collisions_normals_z ((float*)(BASE_ADDRESS + (0x100 * 4 * 4)))


#define CHECKPOINT_ROTATION_SPEED 0.001*3.14159
#define CHECKPOINT_AMPLITUDE 0.01
#define CHECKPOINT_FREQUENCY 0.01

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

    // Clear all collision_mobys
    //for (int i = 0; i < 64; i++) {
    //    collisions_mobys[i] = nullptr;
    //}

    death_count = 0;
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
        seen_planets[9] = 1;
        *(int*)0xa10700 = 1;
        *(int*)0xa10704 = (int)3;  // Kerwan
        //*(int*)0xa10704 = (int)9;  // Gaspar
        //*(int*)0x969c70 = (int)5;
    }

    user_option_camera_rotation_speed = 2;
    user_option_camera_up_down_movement = 0;
    user_option_camera_left_right_movement = 0;

//    if (hoverboard_lady_address == 0) {
//        hoverboard_lady_address = (uint32_t)Moby::find_first(918);
//    } else {
//        Moby* hoverboard_lady = (Moby*)hoverboard_lady_address;
//        if (hoverboard_lady->oClass != 918 || hoverboard_lady->state >= 0x7f) {
//            hoverboard_lady_address = (uint32_t)Moby::find_first(918);
//        }
//    }

//    if (skid_address == 0) {
//        skid_address = (uint32_t)Moby::find_first(717);
//    } else {
//        Moby* skid = (Moby*)skid_address;
//        if (skid->oClass != 717 || skid->state >= 0x7f) {
//            skid_address = (uint32_t)Moby::find_first(717);
//        }
//    }

	if (current_planet == 9 && frame_count < 2) {
        Moby::delete_all(1201);
        Moby::delete_all(1285);

        // Enemies
        Moby::delete_all(193);
        Moby::delete_all(1885);
        Moby::delete_all(1023);
        Moby::delete_all(52);

        //if (death_count <= 0) {
        //    player_pos.z = -10000;
        //}
	}

    if (!current_view) {
        RemoteView* view = new RemoteView();
        this->transition_to(view);
    } else if (current_planet != 0 && frame_count > 2) {
        if (!camera_moby || camera_moby->state >= 0x7f) {
            //camera_moby = Moby::find_first(0x3ef);
        }
        camera_moby = Moby::find_first(0x3ef);

        RemoteView* view = (RemoteView*)current_view;

        // If we're running with renderer, we should render checkpoints
        //if (should_render) {
        if (1) {
            if (checkpoint_moby == nullptr || checkpoint_moby->state >= 0x7f) {
                checkpoint_moby = Moby::spawn(500, 0, 0);
                checkpoint_moby->pUpdate = nullptr;
                //checkpoint_moby->collision = nullptr;
                checkpoint_moby->scale = 0.2f;
                checkpoint_moby->alpha = 64;
            }

//            checkpoint_bounce_z = CHECKPOINT_AMPLITUDE * sin_approx(CHECKPOINT_FREQUENCY * custom_frame_count);
            checkpoint_moby->rotation.z += CHECKPOINT_ROTATION_SPEED;

            checkpoint_moby->position.x = checkpoint_position->x;
            checkpoint_moby->position.y = checkpoint_position->y;
            checkpoint_moby->position.z = checkpoint_position->z;
        }

        // Raycast in a grid pattern extending from the player with the given fov
        // We also oscillate the rays to the left and right to get a wider field of view

        last_death_count = death_count;

        Vec4 forward = camera_forward;
        forward.x = camera_forward.z;
        forward.y = camera_right.z;
        forward.z = camera_up.z;
        forward.w = 1;

        Vec4 left = Vec4();
        left.x = camera_forward.x;
        left.y = camera_right.x;
        left.z = camera_up.x;
        left.w = 1;

        Vec4 up = camera_up;
        up.x = camera_forward.y;
        up.y = camera_right.y;
        up.z = camera_up.y;

        float ray_distance = 64.0f;
        float ray_wide = 90.0f;

        float fov = 64.0f;
        int rows = 16;
        int cols = 16;

        // Store camera collision and set it to null
        unsigned short *collision;
        if (camera_moby != nullptr) {
            collision = camera_moby->collision;
            camera_moby->collision = nullptr;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Vec4 ray = Vec4();
                ray.x = camera_pos.x + ray_distance * forward.x + (ray_wide * (j - cols/2) / cols) * left.x + (ray_wide * (i - rows/2) / rows) * up.x;
                ray.y = camera_pos.y + ray_distance * forward.y + (ray_wide * (j - cols/2) / cols) * left.y + (ray_wide * (i - rows/2) / rows) * up.y;
                ray.z = camera_pos.z + ray_distance * forward.z + (ray_wide * (j - cols/2) / cols) * left.z + (ray_wide * (i - rows/2) / rows) * up.z;
                ray.w = camera_pos.w;

                // To avoid colliding with the camera itself, we start the ray a bit ahead of the camera
                Vec4 ray_start = camera_pos;
                ray_start.x += -1.0f * forward.x;
                ray_start.y += -1.0f * forward.y;
                ray_start.z += -1.0f * forward.z;

                // Apply oscillation
                ray.x += oscillation_offset_x * left.x + oscillation_offset_y * up.x;
                ray.y += oscillation_offset_x * left.y + oscillation_offset_y * up.y;
                ray.z += oscillation_offset_x * left.z + oscillation_offset_y * up.z;

                int coll = coll_line(&ray_start, &ray, 0, nullptr, nullptr);

                Moby* test_moby = nullptr;

                //if (should_render) {
                //    test_moby = (Moby *)collisions_mobys[i * cols + j];
                //}

                if (coll) {
                    collisions_distance[i * cols + j] = distance(camera_pos, coll_output.ip);
                    collisions_class[i * cols + j] = -2 - (coll_output.poly & 0x1fU);
                    collisions_normals_x[i * cols + j] = coll_output.normal.x;
                    collisions_normals_y[i * cols + j] = coll_output.normal.y;
                    collisions_normals_z[i * cols + j] = coll_output.normal.z;

                    if (coll_output.pMoby) {
                        collisions_class[i * cols + j] = coll_output.pMoby->oClass;
                    }

                    //if (should_render) {
                    //    // Update or create moby to show where the raycast hit
                    //    if (test_moby == nullptr || test_moby->state >= 0x7f) {
                    //        test_moby = Moby::spawn(74, 0, 0);
                    //        test_moby->pUpdate = nullptr;
                    //        test_moby->collision = nullptr;
                    //        test_moby->scale = 0.01f;
                    //        test_moby->alpha = 64;
//
                    //        collisions_mobys[i * cols + j] = test_moby;
//
                    //        Logger::debug("Spawning new moby at %f, %f, %f", coll_output.ip.x, coll_output.ip.y,
                    //                      coll_output.ip.z);
                    //    }
//
                    //    test_moby->position.x = coll_output.ip.x;
                    //    test_moby->position.y = coll_output.ip.y;
                    //    test_moby->position.z = coll_output.ip.z;
                    //}
//
//                    Logger::debug("Collided with moby at %f, %f, %f", coll_output.ip.x, coll_output.ip.y, coll_output.ip.z);
                } else {
                    collisions_distance[i * cols + j] = -32.0f;
                    collisions_class[i * cols + j] = -128;

                    //if (should_render && test_moby != nullptr) {
                    //    test_moby->position.z = -100;
                    //}
                }
            }
        }

        // Restore camera collision
        if (camera_moby != nullptr) {
            camera_moby->collision = collision;
        }
    }


    while (progress_frame_count < custom_frame_count && current_planet != 0) {
        sys_ppu_thread_yield();
    }
    custom_frame_count += 1;

    return;
}

void Game::before_player_spawn() {
    death_count += 1;
    checkpoint_moby = nullptr;

    camera_moby = Moby::find_first(0x3ef);

    if (should_render) {
        // Clear collision_mobys
        //for (int i = 0; i < 64; i++) {
        //    collisions_mobys[i] = nullptr;
        //}
    }
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