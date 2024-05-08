# tested with NPEA00387
# need too verify if other versions are different
# loader inject: start->main+78
# substitute (text): seg003 start 0x5F7140 length 0x1618 locals 0x90 args 0x0
# shared data: seg021 length 4

TOC									= 0xc0a4e0 # static throughout
LOADER_INJECT_ADDR 					= 0x9764bc # 0x014D4C # nop in main()
LOADER_START_ADDR 					= 0xa958c0  # 0x20f558 + 4 + align16
LOADER_END_ADDR 					= 0xA959c0 # LOADER_START_ADDR + 0x100
LOADER_SYS_PRX_MODULE_LOAD_ADDR 	= 0xaef624 # cellPrxForUser sys_prx_load_module
LOADER_SYS_PRX_MODULE_START_ADDR 	= 0xaef784 # cellPrxForUser _sys_process_at_Exitspawn 0x96328741

LOADER_SCE_NP_DRM_IS_AVAILABLE_ADDR	= 0xaeef64 # sceNp sceNpDrmIsAvailable2

# 1414 instructions, around ~375 hooks
HOOK_SHARED_TEXT_BEGIN_ADDR = $(LOADER_END_ADDR)
HOOK_SHARED_TEXT_END_ADDR 	= 0xa96ecc	 # HOOK_SHARED_TEXT_BEGIN_ADDR + 0x1588

# 4 bytes
HOOK_SHARED_DATA_BEGIN_ADDR = 0xbd1050
HOOK_SHARED_DATA_END_ADDR 	= 0xbd1054 # HOOK_SHARED_DATA_BEGIN_ADDR + 0x4