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
local x, y, vel_x, vel_y, health, money, bombs, ropes, layer, map_info, powerups, face_left_player, holding_type_player, back_item = 0
local powerups = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} -- 18 powerups, 0 = not, 1 = yes


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

--------------- GAME CONTROL ----------------

-- set_frametime(1/30)
set_speedhack(10)

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

function get_sorted_entities_at(x, y, layer, radius)
    local entities = get_entities_at(0, ~MASK.DECORATION, x, y, layer, radius)
    local distances = {}

    for _, uid in ipairs(entities) do
        local ex, ey = get_position(uid)
        local dx, dy = ex - x, ey - y
        if dx < -17 or dx > 17 or dy < -9 or dy > 9 then --filter out of screen entities
            goto continue
        end
        local dist = dx * dx + dy * dy
        table.insert(distances, {uid = uid, dist = dist})
        ::continue::
    end

    table.sort(distances, function(a, b)
        return a.dist < b.dist
    end)

    local sorted_uids = {}
    for _, entry in ipairs(distances) do
        table.insert(sorted_uids, entry.uid)
    end

    local MAX_ENTITIES = 150
    for i = #sorted_uids, MAX_ENTITIES + 1, -1 do -- remove excess entities
        table.remove(sorted_uids, i)
    end

    info = {}
    for _, uid in ipairs(sorted_uids) do
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

        -- Set some types (enemies, walls, items, etc)
        local type = nil
        local entity_type_flags = entity.type.search_flags
        if bit_extract(entity_type_flags, 0) == 1 then
            type = 1
        elseif bit_extract(entity_type_flags, 2) == 1 then
            type = 2
        elseif bit_extract(entity_type_flags, 3) == 1 then
            type = 3
        elseif bit_extract(entity_type_flags, 5) == 1 then
            type = 4
        elseif bit_extract(entity_type_flags, 8) == 1 then
            type = 5
        elseif bit_extract(entity_type_flags, 15) == 1 then
            type = 6
        elseif bit_extract(entity_type_flags, 9) == 1 then
            type = 7
        end

        table.insert(info,{
            safe(dx, 0), safe(dy, 0),
            safe(vx, 0), safe(vy, 0),
            safe(get_entity_type(uid), 0),
            safe(face_left, -1),
            safe(type, 0),
            safe(holding_type, 0)
        })
    end

    while #info < MAX_ENTITIES do
        table.insert(info, {
            0, 0, 0, 0, -1, -1, 0, 0
        })
    end

    return info
end


-- local function get_entity_info(id, player_x, player_y)
--     local info = {}
--     local entity = get_entity()

--     local x, y, layer = get_position(id)
--     info.x, info.y = x - player_x, y - player_y

--     info.id = id
--     info.type = get_entity_type(id)
--     info.x, info.y = get_position(id)
--     info.layer = get_layer(id)
--     info.health = get_health(id)
--     info.vel_x, info.vel_y = get_velocity(id)
--     info.flags = get_flags(id)
--     info.state = get_state(id)

--     return info
-- end

-- local function get_all_entities_info(player_x, player_y,layer)
--     -- get_entities_at(ENT_TYPE entity_type, int mask, float x, float y, LAYER layer, float radius)
--     local entities = get_entities_at(0, 0, player_x, player_y, layer, 20) -- 20 ~= (17^2 * 9^2)^-2
--     local result = {}
--     for _, id in entities do
--         local x, y, layer = get_position(id)
--         -- filter if it appears on the screen
--         if x < player_x - 17 or x > player_x + 17 or y < player_y - 9 or y > player_y + 9 then
--             goto continue
--         end

--         local info = get_entity_info(id)
--         table.insert(result, info)
--         ::continue::
--     end
--     return result
-- end

-- local function get_map_info(x, y, layer)
--     local left_x, top_y, right_x, bottom_y = get_bounds()
--     local center_x = math.floor(x + 0.5)
--     local start_x, end_x = center_x - 17, center_x + 17
--     local start_y, end_y = math.floor(y - 9), math.ceil(y + 9)

--     -- bounds_data = {
--     --     -- bounds = { left_x = left_x, top_y = top_y, right_x = right_x, bottom_y = bottom_y },
--     --     -- range_x = { start_x = start_x, end_x = end_x },
--     --     -- range_y = { start_y = start_y, end_y = end_y },
--     --     tiles = {}
--     -- }
--     tiles = {}

--     for i = end_y, start_y, -1 do
--         local row = {}
--         for j = start_x, end_x do
--             local tile_id = get_grid_entity_at(j, i, layer)
--             if tile_id == -1 then
--                 -- table.insert(row, { x = i, y = j, type = -1 })
--                 table.insert(row, -1)
--             else
--                 -- table.insert(row, { x = i, y = j, type = get_entity_type(tile_id) })
--                 table.insert(row, get_entity_type(tile_id))
--             end

--         end
--         table.insert(tiles, row)
--     end

--     return tiles
-- end


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

        -- entity_info
        entity_info = get_sorted_entities_at(x,y,layer, 20)   
    else
        health = 0
    end

    local data = {
        -- PLAYER INFO
        -- x = x,
        -- y = y,
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

        -- SCREEN INFO
        world = state.world,
        level = state.level,
        theme = state.theme,
        time = state.time_level,
        win = transition,

        -- ENTITIES INFO
        entities = entity_info
    }

    if transition == 1 then
        transition = 0
    end

    return data
end


--------------- MAIN LOOP ----------------
set_callback(function()
    local level_flags = get_level_flags()
    level_flags = level_flags & ~(1 << 19)
    set_level_flags(level_flags)

    -- IF GAME PAUSED OR LOADING RETURN
    if state.pause ~= 0 or state.loading ~= 0 then
        return
    end
    local line, err = client:receive("*l")
    data = json.decode(line)

    if data["command"] == "reset" then
        reset()
    elseif data["command"] == "step" and #players ~= 0 then
        local python_input = data["input"]
        local last6 = {}
        table.move(python_input, #python_input - 5, #python_input, 1, last6)
        

        local buttons = booleans_to_button_mask(last6)
        steal_input(get_local_players()[1].uid)
        -- x, y go from -1 to 1
        local input = buttons_to_inputs(python_input[1]-1, python_input[2]-1, buttons) -- arrays in lua start at 1
        send_input(players[1].uid, input)
    end

end, ON.PRE_UPDATE)



set_callback(function()
    -- IF GAME PAUSED OR LOADING RETURN
    if state.pause ~= 0 or state.loading ~= 0 then
        return
    end


    if data["command"] == "reset" or data["command"] == "step" then
        serialized_data = json.encode(get_info())
        client:send(serialized_data .. "\n")
    elseif data["command"] == "close" then
        os.exit()
    end

    -- if data["command"] == "step" then
    --     serialized_data = json.encode(get_info())
    --     client:send(serialized_data .. "\n")
    -- end
    
end, ON.POST_UPDATE)



-- WHEN MULTIAGENT, RESET HERE
-- set_callback(function()
-- end, ON.DEATH)

set_callback(function()
    transition = 1
end, ON.TRANSITION)