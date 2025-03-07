local function saveCSV(filename, data)
  local file = io.open(filename, 'w')
  if not file then return false end
  for _, row in ipairs(data) do
    file:write(row.x, ',', row.y, ',', row.z, '\n')
  end
  file:close()
  return true
end

local first = true
local save_path = 'C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/track_extractor/'
function script.windowMain(dt)
  if first then
    first = false

    left = -1
    right = 1
    center = 0

    z=0 -- track progress
    y=0 -- height

    target_boundary = {left, center, right}       -- set your interests
    boundary_name = {'left', 'center', 'right'}   -- column names
    num_points = 5791                             -- number of sampled points

    for target = 1, #target_boundary do
      vec3Data = {}
      x = target_boundary[target]
      for i=0, num_points-1 do
        z = i / num_points
        table.insert(vec3Data, ac.trackCoordinateToWorld(vec3(x,y,z)))
        ac.getTrackAISplineSides(v)
        -- table.insert(vec3Data, ac.getTrackAISplineSides(z))
      end
      saveCSV(save_path .. boundary_name[x+2] .. '.csv', vec3Data)
    end    
  end
end