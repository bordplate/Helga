//
// Created by Vetle Hjelle on 31/12/2022.
//

#ifndef RAC1_MULTIPLAYER_REMOTEVIEW_H
#define RAC1_MULTIPLAYER_REMOTEVIEW_H

#include "../View.h"
#include "../TextElement.h"

#include <lib/vector.h>

struct RemoteView : public View {
    TextElement* get_element(int id);
    void delete_element(int id);

    float coll;
    float coll_up;
    float coll_down;
    float coll_left;
    float coll_right;

    int coll_class;
    int coll_up_class;
    int coll_down_class;
    int coll_left_class;
    int coll_right_class;

    void on_load();
    void render();
private:
    Vector<TextElement*> text_elements_;

    TextElement* ping_text_;
    TextElement* up_text_;
    TextElement* down_text_;
    TextElement* left_text_;
    TextElement* right_text_;
    TextElement* memory_info_text_;
};


#endif //RAC1_MULTIPLAYER_REMOTEVIEW_H
