


def parse_orientation_string(orientation):
    final_vec = []
    split_orientation = orientation.split("-")
    add_negative = False
    for s in split_orientation:
        if len(s) == 0:
            add_negative = True;
        elif add_negative:
            final_vec.append(f'-{s}')
        else:
            final_vec.append(s)
    return final_vec

def extract_pan(orientation):
    return int(parse_orientation_string(orientation)[0])

def extract_tilt(orientation):
    return int(parse_orientation_string(orientation)[1])

def extract_zoom(orientation):
    return int(orientation[-1])

def find_tilt_dist(current_tilt, target_tilt):
    return abs(current_tilt - target_tilt)

def find_pan_dist(current_pan, target_pan):
    if current_pan > target_pan:
        if current_pan - target_pan <= 180:
            # Rotating left
            return current_pan - target_pan
        # Rotating right
        return (360 - current_pan) + target_pan
    else:
        if target_pan - current_pan <= 180:
            # Rotating right
            return target_pan - current_pan
        # Rotating left
        return (360 - target_pan) + current_pan


pan_dist_to_api_encoding = {
    -120: 'FA40',
    -90 : 'FB80',
    -60 : 'FD20',
    -30 : 'FE60',
    0: '0000', 
    30 :  '0140',
    60: '0280',
    90: '0400',
    120: '0520',
}

tilt_dist_to_api_encoding = {
    -30 : 'FE60',
    -15 : 'FF20',
    0: '0000', 
    15 : '0070',
    30 :  '0140',
}

def convert_orientations(center_orientation, infile):
    # For now assume that orientatins don't cross the pan 0 / pan 330 boundary. Center orientations at 180-0-1
    url = "http://128.112.92.59/cgi-bin/ptzctrl.cgi"
    main_pan = extract_pan(center_orientation)
    main_tilt = extract_tilt(center_orientation)

    # Center all orientations at pan 180
    pan_offset = 180 - main_pan
    commands = [ url + "?ptzcmd&abs&24&20&0000&0000" ]
    # Left, middle/up, right, middle/down, middle
    trace = [commands]
    with open(infile, 'r') as f:
        for line in f.readlines():
            line = line.replace('[', '').replace(']', '').replace('\'', '')
            commands = []
            print('new line')
            for o in line.split(','):
                # every orientation is treated as a disposition from center_orientation
                o = o.strip()
                print('orientation ', o)
                pan = extract_pan(o) + pan_offset
                tilt = extract_tilt(o)
                tilt_dist = find_tilt_dist(main_tilt, tilt)
                if tilt < main_tilt:
                    # Tilt downwards since find_tilt_dist doesn't consider direction
                    tilt_dist *= -1
                pan_dist = find_pan_dist(main_pan, pan)
                if pan < main_pan:
                    # Pan left since find_pan_dist doesn't consider direction
                    pan_dist *= -1
                if pan_dist not in pan_dist_to_api_encoding:
                    print(pan_dist , ' pan dist not supported')
                    exit()
                if tilt_dist not in tilt_dist_to_api_encoding:
                    print(tilt_dist , ' tilt dist not supported')
                    exit()
                pan_api_fillin = pan_dist_to_api_encoding[pan_dist]
                tilt_api_fillin = tilt_dist_to_api_encoding[tilt_dist]
                commands.append(f'{url}?ptzcmd&abs&24&20&{pan_api_fillin}&{tilt_api_fillin}')
            trace.append(commands)
    with open('api_trace.txt', 'w') as f:
        for t in trace:
            f.write(str(t).replace('\'', '').replace('[', '').replace(']','') + '\n')
                
                 

def main():
    convert_orientations('120-0-1', 'orientations.txt')


if __name__ == '__main__':
    main()


