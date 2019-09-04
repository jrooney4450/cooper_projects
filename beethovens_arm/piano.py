# `piano.py` - generates coordinates in a csv file for the robot to play. It includes functions to read and write csv files, parse midi files, and define the position of each key on a piano.

# write_coords takes ‘a’, the set of coordinates that you would like to write to a csv file, and saves it as the file ‘name’.

def write_coords(name,a):
    with open(name, mode='w') as coords:
        writer = csv.writer(coords)
        [writer.writerow(i) for i in a]
        return 0

# read_coords takes the file ‘name’ that you would like to read and returns the coordinates that it contains as a numpy array.

def read_coords(name):
    with open(name) as file:
        read = csv.reader(file,delimiter=',')
        e = [q for q in read][::2]
        return np.array([[float(j) for j in row] for row in e])

# parse_song takes the song notes in ‘midi_file’ and looks at the sequence of notes in the song. ‘piano’ is the csv file containing all the coordinates of each key on the piano. This function then looks at the sequence of notes in the song and matches them to the coordinates of those notes on the keyboard. It outputs a numpy array of coordinates on the keyboard for each note in the song in the order that it is in.

def parse_song(midi_file,piano):
    x_black = 5.5
    z_black = 1
    i = np.array(\
        [j.note-28 for j in [msg for msg in mido.MidiFile(midi_file)]\
        [9:-1][::2]])
    a = read_coords(piano)
    b = np.zeros([i.size,3])
    b[:,1] = [a[0,k] for k in i]
    b[:,0] = [x_black * a[1,k] for k in i]
    b[:,2] = [z_black * a[1,k] for k in i]
    return b

# ‘update_coords’ uses measurements on the keyboard and the keyboard’s geometry to create a csv file, ‘name’, with all of the coordinates for each key on the keyboard. When tuning the robot and making it more accurate, it was found that the measurement of some keys were not accurate. To fix this quickly, the function remakes the ‘name’ file with one function call. Furthermore, this function only requires measurements of one octave of notes and then extrapolates measurements for just those 12 notes to the whole keyboard.

def update_coords(name):
    # general measurements 
    dw = 2.3 # cm, white key distance
    doct = 16.15 # cm, octave distance
    
    n = np.zeros(12)
    m = np.ones(12)
    b = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    
    # specific measurements, including black keys along a general octave
    n[0] = 0
    n[1] = 0.9
    n[2] = dw
    n[3] = 3.8
    n[4] = 2*dw
    n[5] = 3*dw
    n[6] = 7.75
    n[7] = 4*dw
    n[8] = 10.45
    n[9] = 5*dw
    n[10] = 13.1
    n[11] = 16.15 - dw
    
    # all the coords on the keyboard
    a = np.zeros([76,2])
    a[:,0] = np.append(np.append(np.append(np.append(np.append(np.append(\
             n[4:]-3*doct*m[4:],n-2*doct*m),n-doct*m),\
             n),n+doct*m),n+2*doct*m),n[:8]+3*doct*m[:8])
    a[:,1] = np.append(np.append(np.append(np.append(np.append(np.append(\
             b[4:],b),b),b),b),b),b[:8])
    return write_coords(name,np.transpose(a))
