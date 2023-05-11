from rosbags.rosbag1 import Reader

bag = Reader.open("./umaVolta.bag")

#print(Reader.topics(bag))
# for topic, msg, t in bag.read_messages(topics=['chatter', 'numbers']):
#     print(msg)
# bag.close()