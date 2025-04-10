meta.unsafe = true

require("os")
package.path = "lua/?.lua;" .. package.path
local socket = require("luasocket.socket")

------------- COMUNICATION ----------------
local port = tonumber(os.getenv("Spelunky_RL_Port"))

local client = socket.tcp()
local success, err = client:connect("127.0.0.1", port)
if not success then
    error("Failed to connect: " .. tostring(err))
end
print("Connected to Python server!")


--------------- GLOBAL VARIABLES ----------------
local data = nil
local transition = 0
local x, y, vel_x, vel_y, health, money, bombs, ropes, layer, map_info, face_left_player, holding_type_player, back_item = 
      0, 0, 0,     0,     0,      0,     0,     0,     0,     0,        0,                 0,                  0
local powerups = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} -- 18 powerups, 0 = not, 1 = yes
local speedup = false

---------------- AUX FUNCTIONS ----------------
local function bit_extract(value, position)
    return math.floor(value / 2^position) % 2
end

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

--------------- GAME CONTROL ----------------


local function reset()
    state.quest_flags = 19
    play_adventure()
    state.items.player_count = 1
    state.items.player_select[1].activated = true
    state.items.player_select[1].character = ENT_TYPE.CHAR_ANA_SPELUNKY
    warp(1, 1, THEME.DWELLING)
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
    -- x: 37.0, y: 118.04999542236, layer: 0
    local left_x, top_y, right_x, bottom_y = get_bounds()
    local start_x, end_x = math.round(x - 10), math.round(x + 10)
    local start_y, end_y = math.round(y - 5), math.round(y + 5)

    tiles = {}
    for i = end_y, start_y, -1 do
        local row = {}
        for j = start_x, end_x do
            local tile_id = get_grid_entity_at(j, i, layer)
            local water_count, lava_count = get_liquids_at(j, i, layer)
            if tile_id == -1 then
                table.insert(row, {0, water_count, lava_count})
            else
                table.insert(row, {get_entity_type(tile_id), water_count, lava_count})
            end
        end
        table.insert(tiles, row)
    end
    return tiles
end


local function get_info()
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

        entity_info = get_entities_info(x,y,layer)
        map_info = get_map_info(x, y, layer)
    else
        health = 0
    end

    local data = {
        player_info = {
            -- x = x,
            -- y = y,
            x_rest  = x - math.floor(x),
            y_rest  = y - math.floor(y),
            -- layer = layer,
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
        },
        
        screen_info = {
            world = state.world,
            level = state.level,
            theme = state.theme,
            time = state.time_level,
            win = transition,
            map_info = map_info
        },

        entity_info = entity_info,
        
    }
    if transition == 1 then
        transition = 0
    end

    return data
end


--------------- MAIN LOOP ----------------
set_callback(function()
    -- FRAMESKIPPING
    if data ~= nil and data["frames"] ~= nil and data["frames"] > 0 then
        return
    end

    -- DISABLE PAUSE
    local level_flags = get_level_flags()
    level_flags = level_flags & ~(1 << 19)
    set_level_flags(level_flags)

    local line, err = client:receive("*l")
    data = json.decode(line)

    if data["command"] == "reset" then
        reset()
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

end, ON.PRE_UPDATE)



set_callback(function()
    
    data["frames"] = data["frames"] - 1
    if data["frames"] <= 0 then

        if data["command"] == "step" then
            serialized_data = json.encode(get_info())
            client:send(serialized_data .. "\n")
        elseif data["command"] == "reset" then
            serialized_data = json.encode(get_info())
            speedup = data["speedup"]
            if speedup then
                set_speedhack(10)
            end
            client:send(serialized_data .. "\n")
        end

    else
        if speedup then
            update_state()
        else
            return
        end
    end    

    

end, ON.POST_UPDATE)



-- WHEN MULTIAGENT, RESET HERE
-- set_callback(function()
-- end, ON.DEATH)

set_callback(function()
    transition = 1
end, ON.TRANSITION)