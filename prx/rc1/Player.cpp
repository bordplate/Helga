//
// Created by Vetle Hjelle on 30/12/2022.
//

#include "Player.h"

#include "Game.h"

void Player::on_tick() {
    if (!ratchet_moby) {
        return;
    }

    last_game_state = game_state;
}

void Player::on_respawned() {

}