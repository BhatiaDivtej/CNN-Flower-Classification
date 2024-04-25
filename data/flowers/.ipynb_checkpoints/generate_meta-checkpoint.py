import os,pdb

f = open('meta/train.txt', 'w')

classes = os.listdir('train')
classes = sorted(classes)
for c in classes:
    print(c)
    class_id = c.split("_")[-1]
    cur_files = os.listdir(os.path.join('train', c))
    for cur_file in cur_files:
        line = c +'/'+ cur_file +' '+ class_id + '\n'
#         pdb.set_trace()
        f.write(line)
f.close()