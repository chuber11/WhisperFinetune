
def get_id_to_nes(nefile):
    id_to_nes = {}
    for line in open(nefile):
        parts = line.strip().split()
        id = parts[0]
        nes = parts[1].split(";")
        id_to_nes[id] = nes
    return id_to_nes

def get_id_to_label(segfiles):
    id_to_label = {}
    for segfile in segfiles:
        for line,line2 in zip(open(segfile),open(segfile.replace("seg.aligned","cased"))):
            id = line.split()[0]
            label = line2.strip()
            id_to_label[id] = label
    return id_to_label

def get_id_to_path(segfiles):
    id_to_path = {}
    for segfile in segfiles:
        for line in open(segfile):
            id, path = line.split()[:2]
            id_to_path[id] = path
    return id_to_path

if __name__ == "__main__":
    segfiles = ["../../WhisperE+Phi2/data/cv.EN.dev.seg.aligned", "../data/voxpopuli.EN.validation.seg.aligned"]
    id_to_label = get_id_to_label(segfiles)

    nefile = "output/dev.txt"
    nes1 = get_id_to_nes(nefile)

    nefile = "output/dev2.txt"
    nes2 = get_id_to_nes(nefile)

    # Analyze differences
    for id in set([*nes1.keys(),*nes2.keys()]):
        nes_1 = set(nes1[id] if id in nes1 else [])
        nes_2 = set(nes2[id] if id in nes2 else [])

        only_in_1 = nes_1 - nes_2
        only_in_2 = nes_2 - nes_1

        if only_in_1 or only_in_2:
            print(f"ID: {id}")
            print(f" Label: {id_to_label[id]}")
            if only_in_1:
                print(f"  Only in dev: {only_in_1}")
            if only_in_2:
                print(f"  Only in dev2: {only_in_2}")
            print()
