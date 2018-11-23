import DisjointSet

default_goals = \
    [False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    True, True, False, False, False, False, False, False, True, True,\
    False, False, False, False, False, False, False, False, False, False,\
    False, False, False, False, False, False, False, False, False, False] 

def analyze_board(board, goals=default_goals, nRows=8, nCols=10, mss=5):
    score_pos = (-1, -1)

    djSet = DisjointSet.DisjointSet(nRows * nCols)

    for row in range(1, nRows):
        for col in range(0, nCols):
            index = row * nCols + col
            up_index = (row-1) * nCols + col
            if (board[index] == board[up_index]):
                setID1 = djSet.getSetID(index)
                setID2 = djSet.getSetID(up_index)
                if (setID1 != setID2):
                    djSet.union(index, up_index)

    for row in range(0, nRows):
        for col in range(1, nCols):
            index = row * nCols + col
            left_index = row * nCols + (col-1)
            if (board[index] == board[left_index]):
                setID1 = djSet.getSetID(index)
                setID2 = djSet.getSetID(left_index)
                if (setID1 != setID2):
                    djSet.union(index, left_index)

    scoringSets = {}
    nearScoringSets = {}
    nonScoringSets = {}


    for row in range(0, nRows):
        for col in range(0, nCols):
            index = row * nCols + col
            s = {}
            s['setID'] = djSet.getSetID(index)
            s['size'] = djSet.getSetSize(index)
            s['color'] = board[index]
            s['rootRow'] = row
            s['rootCol'] = col

            # Goal cells
            if (goals[index] and s['size'] >= mss):
                scoringSets[s['setID']] = s
            elif (goals[index] and s['size'] >= 3):
                nearScoringSets[s['setID']] = s
            else:
                nonScoringSets[s['setID']] = s

    score = 0
    maxSize = 0
    numSets = 0
    numScorableTiles = 0
    scoringSetScore = 0
    nearScoringSetScore = 0
    nonScoringSetScore = 0
    adjacencyScore = 0
    positioningScore = 0

    for s in scoringSets:
        size = scoringSets[s]['size']
        rootRow = scoringSets[s]['rootRow']
        rootCol = scoringSets[s]['rootCol']
        scoringSetScore += pow(size, 2)

        numScorableTiles += size

        if (size > maxSize):
            maxSize = size
            score_pos = (rootRow, rootCol)
        
    for s in nearScoringSets:
        nearScoringSetScore += pow(nearScoringSets[s]['size'], 2)
    
    for s in nonScoringSets:
        nonScoringSetScore += pow(nonScoringSets[s]['size'], 2)

    # Ignore adjacency score and positioning score for now

    numSets = len(scoringSets)

    score = (numSets + numScorableTiles / 3) * 8000 \
        + (scoringSetScore * 1.3 + nearScoringSetScore * 1.3 + nonScoringSetScore * 1.1) * 100 \
        + (adjacencyScore * 20 + positioningScore) * 10


    return score, score_pos

#original C++ code

'''
// analyze the current board and return the "score" associated with it.
// optional parameters return the best row and col to score (the largest score
// set)
int Superball::analyze_superball(int &scoreRow, int &scoreCol)
{
    DJSet = new Disjoint(r*c);
    vector<int> setSizes;
    setSizes.resize(r*c, 1);

    // group all sets of tiles together using a disjoint set

    // union vertically
    for (int row = 1; row < r; row++)
    {
        for (int col = 0; col < c; col++)
        {
            int currentIndex = row * c + col;
            int upIndex = (row-1) * c + col;
            if (board[currentIndex] == board[upIndex])
            {
                int setID1 = DJSet->Find(currentIndex);
                int setID2 = DJSet->Find(upIndex);
                if (setID1 != setID2) 
                {
                    int joinedSet = DJSet->Union(setID1, setID2);
                    if (joinedSet == setID1)
                    {
                        setSizes[setID1] += setSizes[setID2];
                        setSizes[setID2] = 0;
                    }
                    else // joinedSet = setID2
                    {
                        setSizes[setID2] += setSizes[setID1];
                        setSizes[setID1] = 0;
                    }
                }
            }
        }
    }

    // union horizontally
    for (int col = 1; col < c; col++)
    {
        for (int row = 0; row < r; row++)
        {
            int currentIndex = row * c + col;
            int leftIndex = row * c + (col-1);
            if (board[currentIndex] == board[leftIndex])
            {
                int setID1 = DJSet->Find(currentIndex);
                int setID2 = DJSet->Find(leftIndex);
                if (setID1 != setID2) 
                {
                    int joinedSet = DJSet->Union(setID1, setID2);
                    if (joinedSet == setID1)
                    {
                        setSizes[setID1] += setSizes[setID2];
                        setSizes[setID2] = 0;
                    }
                    else // joinedSet = setID2
                    {
                        setSizes[setID2] += setSizes[setID1];
                        setSizes[setID1] = 0;
                    }
                }
            }
        }
    }

    // find and rank all sets of tiles
    map<int, ScoringSet> scoringSets;
    map<int, ScoringSet> nearScoringSets;
    map<int, ScoringSet> nonScoringSets;
    map<int, ScoringSet>::iterator sit;

    for (int row = 0; row < r; row++)
    {
        for (int col = 0; col < c; col++)
        {
            int index = row*c + col;
            // is a non-empty cell
            if (board[index] != '*' && board[index] != '.')
            {
                ScoringSet s;
                s.setID = DJSet->Find(index);
                s.size = setSizes[s.setID];
                s.color = board[index];
                s.rootRow = row;
                s.rootCol = col;
                // goal cells
                if (goals[index] && s.size >= mss)
                    scoringSets.insert(make_pair(s.setID, s));
                // near goal cells
                else if (goals[index] && s.size >= 3)
                {
                    nearScoringSets.insert(make_pair(s.setID, s));
                }
                // all other non goal cells
                else
                    nonScoringSets.insert(make_pair(s.setID, s));
            }
        }
    }


    // calculate grid's "Score"
    int score = 0;
    int maxSize = 0;
    int numSets = 0;
    int numScorableTiles = 0;
    double scoringSetScore = 0;
    double nearScoringSetScore = 0;
    double nonScoringSetScore = 0;
    double adjacencyScore = 0;
    int positioningScore = 0;

    // score based on actual scoring sets and tiles
    for (sit = scoringSets.begin(); sit != scoringSets.end(); sit++)
    {
        ScoringSet s = sit->second;
        
        scoringSetScore += (colors[s.color] + 1 / (colors[s.color] + 3)) * pow(s.size, 2);
        
        numScorableTiles += s.size;
        if (s.size > maxSize)
        {
            maxSize = s.size;
            scoreRow = s.rootRow;
            scoreCol = s.rootCol;
        }
    }

    // near-scoring sets
    for (sit = nearScoringSets.begin(); sit != nearScoringSets.end(); sit++)
    {
        ScoringSet s = sit->second;
        nearScoringSetScore += (colors[s.color] + 1 / (colors[s.color] + 3)) * pow(s.size, 2);
    }
    
    // non scoring sets
    for (sit = nonScoringSets.begin(); sit != nonScoringSets.end(); sit++)
    {
        ScoringSet s = sit->second;
        nonScoringSetScore += (colors[s.color] + 1 / (colors[s.color] + 3)) * pow(s.size, 2);
    }


    // adjacency points (how close same colored tiles are to eachother
    // without actually having to be touching, used mostly for early game)
    for (int i = 2*c; i < board.size()-2-1*c; i++)
    {
        if (board[i] != '*' && board[i] != '.')
        {
            adjacencyScore += adjacent_colors(i, 3);
        }
    }

    // positioning score (for aggregating higher valued tiles into more 
    // favorable tiles on the board that are more likely to be scored, also
    // used mostly for early game)
    for (int i = 0; i < board.size(); i++)
    {
        if (board[i] != '*' && board[i] != '.')
            positioningScore += board[i] * boardValues[i];
    }

    // number of scorable sets bonus
    numSets = scoringSets.size();

    // merge individual scores into a single aggregate score using different
    // weightings on each type of score:
    //
    // For the most part, this formula was derived theoretically by favoring 
    // 1. The number of scoring sets on the board
    // 2. The number of tiles that could be scored
    //    (the plan was to drag out the game as long as possible by continually scoring)
    // 3. The sizes and colors of each scoring set, "near" scoring set (3+ tiles connected
    //    with at least one on a goal tile), and non scoring sets (any other set)
    // 4. Adjacency score (colors being close to, but not necessarily touching, each other),
    //    and positioning score (higher value colors in better spots that are more likely to 
    //    be scored) that are used mostly for early-game situations when there aren't any
    //    potential scoring sets to begin with.
    // 
    // The constants (weightings of each sub score), however, were derived empirically using
    // 11 for-loops worth of brute force, 30 lab machines, and about a week of free 
    // computation time on the hydra machines... >.>)


    score = (numSets + numScorableTiles / 3) * 8000 
        + (scoringSetScore * 1.3 + nearScoringSetScore * 1.3 + nonScoringSetScore * 1.1) * 100
        + (adjacencyScore * 20 + positioningScore) * 10;


    // board not scorable, return -1 for score tile
    if (scoringSets.empty())
    {
        scoreRow = -1;
        scoreCol = -1;
    }

    return score;
}
'''