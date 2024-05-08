#include "rc3/rc3.h"

#ifdef __cplusplus
extern "C" {
#endif

static ModuleInfo sModules[] =
{
    //  short name  long name           toggle setting name     init function       shutdown function       module dependencies
#ifdef GAME_RC3
	{ "ratchetMP", "Ratchet multiplayer mod", "enableRatchetMPModule", rc3_init, rc3_shutdown, NULL },
#endif
};

#ifdef __cplusplus
};
#endif