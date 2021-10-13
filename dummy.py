var = {}

for (p,s,t) in [(1,2,3), (2,3,4), (4,6,2)]:
    if (p,s,t) ==(1,2,3):
        var[(p,s,t)] = 8
    else:
        var[(p, s, t)] = False