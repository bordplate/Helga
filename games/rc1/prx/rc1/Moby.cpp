#include "Moby.h"

#include "Game.h"

#include "rc1.h"

#include <lib/logger.h>

void Moby::set_animation(char animation_id, char animation_frame, u32 duration)
{
    return set_moby_animation(this, animation_id, animation_frame, duration);
}

void Moby::check_collision() {
    if (this->state == 0) {
        this->state = 1;
    }

    MPMobyVars* vars = (MPMobyVars*)(this->pVars);

    if (!vars) {
        return;
    }

    // I'm not sure about these arguments, this kinda works, so I stuck with it.
    int success = coll_sphere(&this->position, this->scale * 10, 0x1, this);

    if (vars->collision_debounce > 0) {
        vars->collision_debounce--;
    }

    if (success > 0 && coll_moby_out != 0 && vars->collision_debounce <= 0) {
        if (coll_moby_out == ratchet_moby) {

        }
    }
}

Moby* Moby::spawn(unsigned short o_class, unsigned short flags, uint16_t modeBits) {
    // If the main hero moby isn't spawned in, we shouldn't try to spawn anything else either.
    if (!ratchet_moby) {
        return nullptr;
    }

    Moby* moby = spawn_moby(o_class);

    if ((int)moby->pVars == nullptr) {
        Logger::error("Moby spawned with invalid pVars: %d. Allocating pVars from custom allocator", (int) moby->pVars);
        moby->pVars = allocate_memory(0x80);
    }

    moby->enabled = 1;
    moby->draw_distance = 0xff;
    moby->update_distance = 0xff;
    moby->alpha = 0xff;

    // Used to set flag 0x1000 here as well, but setting flag 0x1000 makes it so that
    // the first 4 bytes of pVars are checked when using weapons and wrench and that crashes
    // on console when the address doesn't make sense. Does not crash in RPCS3 for whatever reason.
    moby->mode_bits = modeBits;

    Logger::info("Spawned Moby (oClass: %d)", o_class);

    return moby;
}

Moby* Moby::find_first(unsigned short o_class) {
    for (Moby *moby = moby_ptr; moby <= moby_ptr_end; moby++) {
        if (moby->state < 0x7f && moby->oClass == o_class) {
            return moby;
        }
    }

    return nullptr;
}

Moby* Moby::find_last(unsigned short o_class) {
    Moby *last;

    for (Moby *moby = moby_ptr; moby <= moby_ptr_end; moby++) {
        if (moby->state < 0x7f && moby->oClass == o_class) {
            Logger::debug("Found moby with oClass: %d", moby->oClass);
            last = moby;
        }
    }

    return last;
}