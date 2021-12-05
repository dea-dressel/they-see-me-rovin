import random as r

NUM_ROWS = 10
NUM_COLS = 10

R_MINERAL = 20
R_WALL = -10
R_HOLE = -20
R_CLEAR = -1

# WEST  == 0; EAST  == 1; NORTH == 2; SOUTH == 3
ACTION_DICT = {0:-1, 1:1, 2:-NUM_ROWS, 3:NUM_ROWS}


#------------------------------------------------------------------
'''
Generates a set of hole locations for our 10x10 grid.
Must be run before get_mineral_locations to avoid 
putting minerals and holes in the same location.
'''
def get_mineral_locations(num_minerals):
  result = set()

  while len(result) < num_minerals:
    rand_location = r.randint(1,NUM_ROWS*NUM_COLS)
    result.add(rand_location)

  return result

#------------------------------------------------------------------
def get_next_state(state, action, did_hit_wall):
  if did_hit_wall:
    return state
  else:
    return state + ACTION_DICT[action]

#------------------------------------------------------------------
'''
Given a state/action pair, we return true if we hit a wall.
'''
def hit_wall(state, action):
  west_hit = state % NUM_COLS == 1 and action == 0
  east_hit = state % NUM_COLS == 0 and action == 1
  north_hit = state in range(1,NUM_COLS+1) and action == 2
  south_hit = state in range(NUM_ROWS*NUM_COLS - NUM_COLS, (NUM_ROWS*NUM_COLS)+1) and action == 3

  return west_hit or east_hit or north_hit or south_hit
#------------------------------------------------------------------

'''
Generates a set of hole locations for our 10x10 grid.
Must be run after get_hole_locations to avoid 
putting minerals and holes in the same location.
'''
def get_hole_locations(num_holes, minerals):
  result = set()
  while len(result) < num_holes:
    rand_location = r.randint(1,NUM_ROWS * NUM_COLS)
    if rand_location not in minerals:
      result.add(rand_location)
  return result
#------------------------------------------------------------------

def get_reward(state, action, hit_wall, next_state, minerals, holes):
  #If we hit a wall...
  if hit_wall:
    return R_WALL
  
  #If we fall into a hole...
  if next_state in holes:
    return R_HOLE

  #If we find a mineral...
  if next_state in minerals:
    return R_MINERAL

  return R_CLEAR

#------------------------------------------------------------------
'''
Given a set of mineral locations and hole locations,
we return a list of (s,a,r,sp) tuples
'''
def get_lines(minerals, holes):
  result = []
  for i in range(1, (NUM_ROWS * NUM_COLS) + 1):
    for action in ACTION_DICT:
      state = i
      action = action
      did_hit_wall = hit_wall(state, action)
      next_state = get_next_state(state, action, did_hit_wall)
      reward = get_reward(state, action, did_hit_wall, next_state, minerals, holes)
      line = (state, action, reward, next_state)
      result.append(line)
  return result

#------------------------------------------------------------------
'''
Takes a list of tupes `csv_lines` and writes
those tuples into a csv format
'''
def write_csv(minerals, holes, filename, csv_lines):
  m_list = list(minerals)
  h_list = list(holes)
  with open(filename, 'w') as f:
    for i in range(len(m_list) - 1):
      f.write(str(m_list[i]) + ',')
    f.write(str(m_list[-1]) + '\n')

    for i in range(len(h_list) - 1):
      f.write(str(h_list[i]) + ',')
    f.write(str(h_list[-1]) + '\n')

    f.write(str(NUM_ROWS*NUM_COLS) + '\n')
    f.write(str(len(ACTION_DICT)) + '\n')
    f.write("s,a,r,sp\n")
    for s, a, r, sp in csv_lines:
      f.write(str(s) + ',' + str(a) + ',' + str(r) + ',' + str(sp) + '\n')
#------------------------------------------------------------------

def get_grid(minerals, holes):
  index = 1
  result = []
  while index <= NUM_ROWS*NUM_COLS:
    row = []
    for i in range(NUM_COLS):
      if index in minerals:
        row.append(R_MINERAL)
      elif index in holes:
        row.append(R_HOLE)
      else:
        row.append(R_CLEAR)
      index += 1
    result.append(row)
  return result

#------------------------------------------------------------------

'''
Randomly Creates a CSV that represents a 10x10 grid in a SARS' format.
The top left entry of the grid is labeled "1" 
The bottom right entry of the grid is labeled "100"

-----------------------
  CSV Format: 
    num_minerals
    num_holes
    num_states
    num_actions
    S1,A1,R,S'
    S2,A2,R,S'
    ...
    Sn,An,R,S'

'''
def generateRandomCSV(num_minerals, num_holes, filename):

  minerals = get_mineral_locations(num_minerals)
  holes = get_hole_locations(num_holes, minerals)
  csv_lines = get_lines(minerals, holes)
  write_csv(minerals, holes, filename, csv_lines)
  grid = get_grid(minerals, holes)
  #print("mineral locations: " + str(minerals))
  #print("hole locations: " + str(holes))
  #print()
  #return grid

#------------------------------------------------------------------

'''
User specifies the locations of minerals and holes as integers.
A csv is then generated based on these locations
'''
def generateFixedCSV(minerals, holes, filename):
  csv_lines = get_lines(minerals, holes)
  write_csv(minerals, holes, filename, csv_lines)
  grid = get_grid(minerals, holes)
  print("mineral locations: " + str(minerals))
  print("hole locations: " + str(holes))

def main(num_grids):
  for i in range(num_grids):
    filename = "./grids/test_" + str(i) + ".csv"
    num_minerals = r.randint(1,25)
    num_holes = r.randint(1,25)
    generateRandomCSV(num_minerals,num_holes,filename)

main(1000)