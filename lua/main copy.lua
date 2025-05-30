meta.unsafe = true
require("os")

------------- COMUNICATION ----------------
package.path = "lua/?.lua;" .. package.path
local socket = require("luasocket.socket")
local port = tonumber(os.getenv("Spelunky_RL_Port"))

local client = socket.tcp()
local success, err = client:connect("127.0.0.1", port)
if not success then
    error("Failed to connect: " .. tostring(err))
end
print("Connected to Python server!")


--------------- GLOBAL VARIABLES ----------------
local transition = 0
local speedup = false
local speedup_counter = 100
local data = {
    frames = 0,
    command = "pass"
}

local x, y, vel_x, vel_y, health, money, bombs, ropes, layer, map_info, face_left_player, holding_type_player, back_item, dist_to_goal, pos_type_matrix, char_state, can_jump = 
      0, 0, 0,     0,     0,      0,     0,     0,     0,     0,        0,                 0,                  0,         0,            0,               0,          false
local powerups = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} -- 18 powerups, 0 = not, 1 = yes

---------------- PATHFINDING ----------------
local Grid        = require ("jumper.grid")
local Pathfinder  = require ("jumper.pathfinder")

local pf_grid       = nil        -- jumper grid
local pf_finder     = nil        -- jumper finder
local pf_board      = {}         -- 0 = walkable, 1 = blocked
local pf_ntiles     = 0          -- rebuild only if the number of floor tiles changes
local pf_xmin, pf_ymin, pf_xmax, pf_ymax = 0,0,0,0
local pf_goalx, pf_goaly = 0,0   -- exit cell coords
local pf_dirty = true

---------------- AUX FUNCTIONS ----------------
local function safe(val, default)
    if val ~= nil and val then
        return val
    else
        return default
    end
end

function math.round(x)
    return math.floor(x + 0.5)
end

function destroy_entities(entity_types)
    if #entity_types == 0 then
        return
    end
    for _, uid in ipairs(get_entities_by_type(entity_types)) do
        kill_entity(uid)
    end
end

-- Build / refresh the grid
local function pf_refresh()
    local tiles = get_entities_by(0, MASK.FLOOR, 0)
    if #tiles == pf_ntiles then return end

    -- Re‑compute bounds & blank board
    pf_ntiles      = #tiles
    pf_xmin,pf_ymin,pf_xmax,pf_ymax = get_bounds()
    pf_board       = {}
    for y = pf_ymin - 0.5, pf_ymax + 0.5, -1 do
        local row = {}
        for x = pf_xmin + 0.5, pf_xmax - 0.5, 1 do row[#row+1] = 0 end
        pf_board[#pf_board+1] = row
    end

    -- Mark blocking tiles
    for _,uid in ipairs(tiles) do
        local e = get_entity(uid)
        if (test_flag(e.flags,3) and e.type.id ~= ENT_TYPE.FLOOR_PIPE)
           or e.type.id == ENT_TYPE.FLOOR_SPIKES then
            local tx = math.floor( e.x - pf_xmin + 1 )
            local ty = math.floor( pf_ymin - e.y + 1 )
            if pf_board[ty] and pf_board[ty][tx] then pf_board[ty][tx] = 1 end
        end
    end

    -- Goal (first exit door we see)
    local exits = get_entities_by_type(ENT_TYPE.FLOOR_DOOR_EXIT)
    if #exits > 0 then
        local gx,gy = get_position(exits[1])
        pf_goalx = math.floor( gx - pf_xmin + 1 )
        pf_goaly = math.floor( pf_ymin - gy + 1 )
    end

    pf_grid   = Grid(pf_board)
    pf_finder = Pathfinder(pf_grid,"ASTAR",0)
end


-- Return path length (-1 if no path / no exit yet)
local function pf_distance(px,py)
    if pf_dirty then
        pf_refresh()
        pf_dirty = false
    end
    local fx = math.floor( px - pf_xmin + 1 )
    local fy = math.floor( pf_ymin - py + 1 )
    local path = pf_finder and pf_finder:getPath(fx,fy,pf_goalx,pf_goaly) or nil
    local path_length = -1
    if path then
        path_length = path:getLength() - 1
    end
    return path_length
end

local function set_pf_dirty()
    pf_dirty = true
    print("Refreshed")
end

set_post_entity_spawn(function(ent)
    ent:set_pre_destroy(function(self)
        set_pf_dirty()
    end)
    pf_dirty = true
    print("Refreshed")
end, SPAWN_TYPE.ANY,MASK.FLOOR)

--------------- GAME CONTROL ----------------

local function reset(seed)
    set_adventure_seed(seed, seed)
    state.items.player_count = 1
    state.items.player_select[1].activated = true
    state.items.player_select[1].character = ENT_TYPE.CHAR_ANA_SPELUNKY
    warp(1, 1, THEME.DWELLING)
    
    god(true)
end

local button_map = {
    BUTTON.JUMP,
    BUTTON.WHIP,
    BUTTON.BOMB,
    BUTTON.ROPE,
    BUTTON.RUN,
    BUTTON.DOOR,
}

local function booleans_to_button_mask(booleans)
    local mask = 0
    for i, pressed in ipairs(booleans) do
        if pressed == 1 then
            mask = mask | button_map[i]
        end
    end
    return mask
end


--------------- INFO RETRIEVAL ----------------

function get_entities_info(x, y, layer)
    mask = 0xFFFFFFFF & ~(MASK.DECORATION | MASK.BG | MASK.SHADOW | MASK.FLOOR | MASK.LIQUID | MASK.FX)
    local entities = get_entities_overlapping_hitbox(
        0, -- all types of entity
        mask,
        AABB:new(math.round(x-10), math.round(y+5), math.round(x+10), math.round(y-5)),
        layer
    )
    info = {}
    for _, uid in ipairs(entities) do
        local entity = get_entity(uid)
        local ex, ey = get_position(uid)
        local dx, dy = ex - x, ey - y
        local vx, vy = get_velocity(uid)
        local face_left = (entity.flags & (1 << 16)) ~= 0
        
        local holding_type = 0
        if entity.holding_uid ~= -1 and entity.holding_uid ~=0 and entity.holding_uid ~= nil then
            holding_type = get_entity_type(entity.holding_uid)
        end
        -- local back_type = 0
        -- if worn_backitem(uid) ~= -1 then
        -- if entity:worn_backitem() ~= -1 then
            -- local back_entity = get_entity(worn_backitem(uid))
            -- if back_entity then
            --     back_type = get_entity_type(back_entity)
            -- end
        -- end

        local type = entity.type.search_flags

        table.insert(info,{
            safe(dx, 0), safe(dy, 0),
            safe(vx, 0), safe(vy, 0),
            safe(get_entity_type(uid), 0),
            safe(face_left, -1),
            safe(holding_type, 0),
        })
    end

    return info
end

local function get_map_info(x, y, layer)
    local start_x, end_x = math.round(x - 10), math.round(x + 10)
    local start_y, end_y = math.round(y - 5), math.round(y + 5)

    tiles = {}
    -- pos_type_matrix = {}
    for i = end_y, start_y, -1 do
        local row = {}
        -- local pos_type_row = {}
        for j = start_x, end_x do
            local tile_id = get_grid_entity_at(j, i, layer)
            local water_count, lava_count = get_liquids_at(j, i, layer)

            -- pos_type
            -- local pos_type_flags = {}
            -- local keys = {}
            -- for k in pairs(POS_TYPE) do keys[#keys+1] = k end
            -- table.sort(keys)
            -- for _, k in ipairs(keys) do
            --     pos_type_flags[#pos_type_flags+1] = position_is_valid(j, i, layer, POS_TYPE[k]) and 1 or 0
            -- end
            -- table.insert(pos_type_row, pos_type_flags)
            --

            if tile_id == -1 then
                table.insert(row, {0, water_count, lava_count})
            else
                table.insert(row, {get_entity_type(tile_id), water_count, lava_count})
            end
        end
        table.insert(tiles, row)
        -- table.insert(pos_type_matrix, pos_type_row)
    end
    return tiles --, pos_type_matrix
end


local function get_info(additional_fields)
    if #players ~= 0 then
        x, y, layer = get_position(players[1].uid)
        health, money, bombs, ropes = players[1].health, players[1].inventory.money, players[1].inventory.bombs, players[1].inventory.ropes
        
        vel_x, vel_y = get_velocity(players[1].uid)
        face_left_player = (players[1].flags & (1 << 16)) ~= 0
        holding_type_player = 0
        if players[1].holding_uid ~= -1 and players[1].holding_uid ~=0 and players[1].holding_uid ~= nil then
            holding_type_player = get_entity_type(players[1].holding_uid)
        end
        back_item = 0
        if players[1]:worn_backitem() ~= -1 and players[1]:worn_backitem() ~= 0 and players[1]:worn_backitem() ~= nil then
            back_item = get_entity_type(players[1]:worn_backitem())
        end

        powerups = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
        for _, value in ipairs(players[1]:get_powerups()) do
            if value ~= 0 then
                powerups[value-545+1] = 1 
            end
        end

        char_state = players[1].state
        can_jump = players[1]:can_jump()

    else
        health = 0
    end

    local data = {
        basic_info = {
            x = x,
            y = y,
            x_rest  = x - math.floor(x),
            y_rest  = y - math.floor(y),
            layer = layer,
            health = health,
            bombs = bombs,
            ropes = ropes,
            money = money,
            vel_x = vel_x,
            vel_y = vel_y,
            face_left = face_left_player,
            powerups = powerups,
            holding_type_player = holding_type_player,
            back_item = back_item,
            char_state = char_state,
            can_jump = can_jump,
            world = state.world,
            level = state.level,
            theme = state.theme,
            time = state.time_level,
            win = transition,
        },
    }
    if transition == 1 then
        transition = 0
    end

    -- ADDITIONAL INFO
    for i, value in ipairs(additional_fields) do
        if value == "map_info" then
            data.map_info = get_map_info(x, y, layer)
        elseif value == "dist_to_goal" then
            data.dist_to_goal = pf_distance(x,y)
        elseif value == "entity_info" then
            data.entity_info = get_entities_info(x,y,layer)
        elseif value == "custom_info" then
            data.custom_info = ""
        end
    end

    return data
end


set_callback(function()    
    data["frames"] = data["frames"] - 1
    if data["frames"] <= 0 then

        local start = get_performance_counter()
        -- SEND
        if data["command"] == "step" then
            serialized_data = json.encode(get_info(data["additional_data"]))
            local finish = get_performance_counter()
            local freq = get_performance_frequency()
            local elapsed_time = (finish - start) / freq
            -- print(string.format("Get info: %.6f seconds", elapsed_time))
            start = get_performance_counter()
            client:send(serialized_data .. "\n")
        elseif data["command"] == "reset" then
            destroy_entities(data["ent_types_to_destroy"])
            serialized_data = json.encode(get_info(data["additional_data"]))
            speedup = data["speedup"]
            if speedup then
                set_speedhack(100)
            end
            client:send(serialized_data .. "\n")
        end

        -- RECEIVE
        local line, err = client:receive("*l")
        data = json.decode(line)
        local finish = get_performance_counter()
        local freq = get_performance_frequency()
        local elapsed_time = (finish - start) / freq
        -- print(string.format("Elapsed time: %.6f seconds", elapsed_time))


        if data["command"] == "reset" then
            reset(data["seed"])
            data["frames"] = 60
        elseif data["command"] == "step" and #players ~= 0 then
            local python_input = data["input"]
            local last6 = {}
            table.move(python_input, #python_input - 5, #python_input, 1, last6)
            local buttons = booleans_to_button_mask(last6)

            steal_input(get_local_players()[1].uid)
            -- x, y go from -1 to 1
            local input = buttons_to_inputs(python_input[1]-1, python_input[2]-1, buttons) -- arrays in lua start at 1
            send_input(players[1].uid, input)

        elseif data["command"] == "close" then
            os.exit()
        end

    end 
    if speedup then
        speedup_counter = speedup_counter - 1
        if speedup_counter <= 0 then
            speedup_counter = 150
            return
        end
        update_state()
    end

    -- DISABLE PAUSE
    local level_flags = get_level_flags()
    level_flags = level_flags & ~(1 << 19)
    set_level_flags(level_flags)
end, ON.POST_UPDATE)

set_callback(function()
    transition = 1
end, ON.TRANSITION)