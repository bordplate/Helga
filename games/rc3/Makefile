-include userconfig.mk
-include os.mk

# dirs
BASE_DIR = $(CURDIR)

# fix path separator issues
PS3_SDK := $(shell echo $(PS3_SDK) | sed 's:\\\\:/:g')
BASE_DIR := $(shell echo $(BASE_DIR) | sed 's:\\\\:/:g')
RPCS3_DIR := $(shell echo $(RPCS3_DIR) | sed 's:\\\\:/:g')

TOOLS_DIR = $(BASE_DIR)/tools

BIN_DIR = $(BASE_DIR)/build

LOADER_DIR = $(BASE_DIR)/loader
LOADER_BUILD_DIR = $(LOADER_DIR)/build

APP_DIR = $(BASE_DIR)/app
APP_BUILD_DIR = $(APP_DIR)/build

PRX_DIR = $(BASE_DIR)/prx
PRX_BUILD_DIR = $(PRX_DIR)/build
PRX_OUT_DIR = $(PRX_BUILD_DIR)/tmp
PRX_BUILD_IN_DIR = $(PRX_BUILD_DIR)/tmp
PRX_BUILD_TMP_DIR = $(PRX_BUILD_DIR)/tmp
PRX_BUILD_OUT_DIR = $(PRX_BUILD_DIR)/bin
BIN2RPCS3PATCH = $(TOOLS_DIR)/bin2rpcs3patch.py
GENERATEPATCHFILE = $(TOOLS_DIR)/generate_patch.py
PKG_NPDRM = $(WINE) $(PS3_SDK)/Tools/psn_package_npdrm.exe

# include game specific makefile settings
include config_$(GAME).mk

PATCH_FILE ?= $(RPCS3_DIR)/patches/patch.yml

# handle GAME/DISC category
ifeq ($(GAME_CAT), HDD)
GAME_DIR ?= $(RPCS3_DIR)/dev_hdd0/game/$(GAME_ID)/USRDIR
else
GAME_DIR ?= $(RPCS3_DIR)/dev_hdd0/disc/$(GAME_ID)/PS3_GAME/USRDIR
endif

# merge user specified hooks files with the game specific one
HOOKS_FILES := $(HOOKS_FILES) $(PRX_DIR)/$(GAME_FOR_PATHS)/hooks.yml

# workaround for hooks argument not accepting an empty list
ifneq ($(HOOKS),)
HOOKSARG = --hooks $(HOOKS)
endif

BIN2RPCS3PATCHARGS = \
	--input "$(LOADER_BUILD_DIR)/loader.text.inject.bin" "$(LOADER_BUILD_DIR)/loader.text.bin" --address $(LOADER_INJECT_ADDR) $(LOADER_START_ADDR) \
	--output "$(PATCH_FILE)" --indent 3 --replace_patch shk_elf_loader_$(GAME)

GENERATEPATCHFILEARGS = \
	--input "$(LOADER_BUILD_DIR)/loader.text.inject.bin" "$(LOADER_BUILD_DIR)/loader.text.bin" --address $(LOADER_INJECT_ADDR) $(LOADER_START_ADDR) \
	--output "$(LOADER_BUILD_DIR)/patch.txt" --append

SHKGENARGS = \
	--tools_dir "$(TOOLS_DIR)" --elf_out_dir "$(LOADER_BUILD_DIR)" --prx_out_dir "$(PRX_OUT_DIR)" \
	--toc $(TOC) --hook_shared_text_range $(HOOK_SHARED_TEXT_BEGIN_ADDR) $(HOOK_SHARED_TEXT_END_ADDR) \
	--hook_shared_data_range $(HOOK_SHARED_DATA_BEGIN_ADDR) $(HOOK_SHARED_DATA_END_ADDR) --patch_file "$(PATCH_FILE)" $(HOOKSARG) --hooks_file $(HOOKS_FILES) \
	--game $(GAME) --loader_inject_addr $(LOADER_INJECT_ADDR) --loader_text_range $(LOADER_START_ADDR) $(LOADER_END_ADDR) \
	--sys_prx_load_module_addr $(LOADER_SYS_PRX_MODULE_LOAD_ADDR) --sys_prx_start_module_addr $(LOADER_SYS_PRX_MODULE_START_ADDR) --sce_np_drm_is_available2_addr $(LOADER_SCE_NP_DRM_IS_AVAILABLE_ADDR)

all:
# generate build folders
	mkdir -p loader/build
	mkdir -p prx/build/tmp
	mkdir -p prx/build/bin

# generate build files
	cd tools && $(PYTHON) shkgen.py $(SHKGENARGS)

# base game patch
	cp -f "basepatches/rc3.txt" "$(LOADER_BUILD_DIR)/patch.txt"

# build loader
	cd "$(LOADER_DIR)" && "$(MAKE)" binary LOADER_INJECT_ADDR=$(LOADER_INJECT_ADDR) LOADER_START_ADDR=$(LOADER_START_ADDR)
	#$(PYTHON) "$(BIN2RPCS3PATCH)" $(BIN2RPCS3PATCHARGS)
	$(PYTHON) "$(GENERATEPATCHFILE)" $(GENERATEPATCHFILEARGS)

# build injection patch
	cd "$(LOADER_BUILD_DIR)" && "$(MAKE)" -f shk_elf.gen.mk patch

# patch eboot
	cd "$(LOADER_DIR)" && "$(MAKE)" patch

# build sprx
	cd "$(PRX_DIR)" && "$(MAKE)" sprx GAME=$(GAME)
	cp "$(PRX_BUILD_OUT_DIR)/mod.sprx" "$(GAME_DIR)"

# make launcher app
	cd "$(APP_DIR)" && "$(MAKE)" eboot

# put in build folder
	rm -rf $(BIN_DIR)/PS3_GAME/
	mkdir -p $(BIN_DIR)/PS3_GAME/USRDIR
	cp -f "$(APP_DIR)/ICON0.png" "$(BIN_DIR)/PS3_GAME/ICON0.png"
	cp -f "$(APP_DIR)/PARAM.SFO" "$(BIN_DIR)/PS3_GAME/PARAM.SFO"
	cp -f "$(APP_BUILD_DIR)/bin/EBOOT.BIN" "$(BIN_DIR)/PS3_GAME/USRDIR/EBOOT.BIN"
	cp -f "$(LOADER_BUILD_DIR)/EBOOT.mp.BIN" "$(BIN_DIR)/PS3_GAME/USRDIR/EBOOT.mp.BIN"
	cp -f "$(PRX_BUILD_DIR)/bin/mod.sprx" "$(BIN_DIR)/PS3_GAME/USRDIR/mod.sprx"

# build package
	cd $(PS3_SDK)/Tools && \
	$(PKG_NPDRM) $(CURDIR)/package.conf $(BIN_DIR)/PS3_GAME && \
	cp BDUPS3-BORD00001_00-0000000000000000.pkg $(BIN_DIR)/

sprx:
	cd "$(PRX_DIR)" && "$(MAKE)" sprx GAME=$(GAME)
	cp "$(PRX_BUILD_OUT_DIR)/mod.sprx" "$(BIN_DIR)/PS3_GAME/USRDIR/mod.sprx"

clean:
	cd "$(LOADER_DIR)" && "$(MAKE)" clean
	cd "$(PRX_DIR)" && "$(MAKE)" clean

setup:
# make sure GAME variable is set for initial setup
#	ifeq ($(GAME),)
#	$(error Need to specify GAME variable for initial setup)
#	endif

# copy userconfig from template
	-$(COPY) userconfig.mk userconfig.mk.bak
	-$(COPY) userconfig_$(GAME).template.mk userconfig.mk

# create folders used during build
	-mkdir "$(LOADER_BUILD_DIR)"
	-mkdir "$(PRX_BUILD_DIR)"
	-mkdir "$(PRX_BUILD_IN_DIR)"
	-mkdir "$(PRX_BUILD_TMP_DIR)" 
	-mkdir "$(PRX_BUILD_OUT_DIR)"


