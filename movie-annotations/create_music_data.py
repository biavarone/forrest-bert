import codecs

path_to_emo_data = '../../FORREST/to_bert/emo/sub-04_2sec.tsv'
path_to_music_data = '../../../ownCloud/Dottorandi/Benedetta/FORREST/DATA/music.csv'

with codecs.open(path_to_emo_data, 'r', 'utf-8') as emo_file:
    with codecs.open(path_to_music_data, 'r', 'utf-8') as music_file:
        timestamps = []
        music_timestamps = []

        for line in emo_file:
            line = line.strip().split('\t')
            timestamps.append(float(line[0]))

        next(music_file)
        for line in music_file:
            line = line.strip().split(',')
            start_end = [float(line[0]), float(line[1])]
            music_timestamps.append(start_end)

        music_presence = {}
        outfile = codecs.open('data/music/music_6sec.tsv', 'w', 'utf-8')

        # one hot encoding for music [music, no music]
        for ts in timestamps:
            for i in range(int(ts)-6, int(ts)+1):
                for m_ts in music_timestamps:
                    if m_ts[0] <= i <= m_ts[1]:
                        music_presence[ts] = [1, 0]
                    else:
                        if ts not in music_presence:
                            music_presence[ts] = [0, 1]

        for key, value in music_presence.items():
            outfile.write(str(int(key)) + '\t' + str(value[0]) + '\t' + str(value[1]) + '\n')


