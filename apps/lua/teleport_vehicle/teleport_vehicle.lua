---@diagnostic disable: undefined-global, undefined-field, missing-parameter, need-check-nil, missing-return, param-type-mismatch, lowercase-global, redundant-value, cast-local-type,duplicate-set-field

local PREFIX_teleport = 'C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/teleport_vehicle/teleport_point.csv'

local owncar = ac.getCar(0)
local vec = {x=vec3(1,0,0),y=vec3(0,1,0),z=vec3(0,0,1),empty=vec3()}
local pos2, pos3, rot3 = vec2(), vec3(), vec3()
local allow = true
local pre_data = {}

function readCSV(file)
  local file = io.open(file, "r")
  if not file then return nil, "File not found" end
  local data = {}
  for line in file:lines() do
    local temp = {}
    for word in string.gmatch(line, '([^,]+)') do
      table.insert(temp, word)
    end
    table.insert(data, temp)
  end
  file:close()
  return data
end

function tablesEqual(t1, t2)
  if type(t1) ~= type(t2) then return false end

  if type(t1) ~= 'table' then return t1 == t2 end

  for k, v in pairs(t1) do
      if not tablesEqual(v, t2[k]) then
          return false
      end
  end

  for k in pairs(t2) do
      if t1[k] == nil then
          return false
      end
  end

  return true
end

local cnt = 0
function script.windowMain(dt)
  allow = true

  local data, err = readCSV(PREFIX_teleport)
  -- print(data)
  -- print(pre_data)
  if tablesEqual(data, pre_data) then return end
  
  pre_data = data
  pos2:set(tonumber(data[1][1]), tonumber(data[1][2]))
  rot3:set(tonumber(data[1][3]), 0, tonumber(data[1][4]))
  local raycastheight = 3000
  pos3:set(pos2.x, raycastheight, pos2.y)
  local initialray = physics.raycastTrack(pos3,-vec.y,raycastheight*2)

  local normalize3Dto2Dto3D = function (vector)
    local temp = vec2(vector.x,vector.z):normalize()
    return vec3(temp.x,0,temp.y)
  end
  local carside = normalize3Dto2Dto3D(owncar.side)
  local carlook = normalize3Dto2Dto3D(owncar.look)
  for i=1, 100 do
    local side = math.random(-owncar.aabbSize.x/2 +owncar.aabbCenter.x,owncar.aabbSize.x/2 +owncar.aabbCenter.x)
    local look = math.random(-owncar.aabbSize.z/2 +owncar.aabbCenter.z,owncar.aabbSize.z/2 +owncar.aabbCenter.z)
    local pos = pos3 + carlook*look + carside*side
    local raycastnormal = vec3()
    local raycast = physics.raycastTrack(pos, -vec.y, raycastheight*2, _, raycastnormal)
    if raycast == -1 or math.abs(raycast-initialray)>0.2 then allow = false
    end
  end
  pos3:set(pos2.x, raycastheight-initialray+3, pos2.y)

  if allow then
    if owncar.physicsAvailable then
      -- print("Teleport Vehicle", cnt)
      cnt = cnt + 1
      -- physics.setCarPosition(0,pos3,-owncar.look)
      physics.setCarPosition(0,pos3,-rot3)      
    end
  end
end

