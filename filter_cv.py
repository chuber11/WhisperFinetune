
from glob import glob

def use(label):
    label = label.lower()
    for term in ["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fiveteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","fourty","fivety","sixty","seventy","eighty","ninety","hundred","thousand","million","eins","zwei","drei","vier","fünf","sechs","sieben","acht","neun","zehn","elf","zwölf","dreizehn","vierzehn","fünfzehn","sechzehn","siebzehn","achtzehn","neunzehn","zwanzig","dreißig","vierzig","fünfzig","sechzig","siebzig","achtzig","neunzig","hundert","tausend","million"]:
        if term in label:
            return False
    return True

for path in ["data/cv.*.*.seg.aligned","data_test/cv.*.test.seg.aligned"]:
    for segfile in glob(path):
        labelfile = segfile.replace(".seg.aligned",".cased")
        segfile_out = segfile.replace("data","data_filtered").replace("cv","cv_filtered")
        labelfile_out = segfile_out.replace(".seg.aligned",".cased")
        print(segfile_out)

        with open(segfile_out, "w") as f, open(labelfile_out,"w") as f2:
            for line, line2 in zip(open(segfile),open(labelfile)):
                label = line2.strip()

                if use(label):
                    f.write(line)
                    f2.write(line2)
