# Requirements

Install following requirements
```
numpy
scipy
scikit-learn
pandas
sentence_transformers
BERTopic
tqdm
```

# Instructions
1. Place the AOL log files in a directory named "AOL".
2. Run ```parse_aol_log.py``` file, which does the session segmentations of the AOL log file and produces ```new_logs_*.csv``` file.
3. Run ```create_keyword_clusters.py``` file, which samples a subset of sessions from the previous stage, creates User Query clusters with search engine performance metrics per session, and produce ```sequences-50000-*.json``` file.  
4. Run ```create_behavior_cluster.py``` file, which assigns behavior cluster to each session in each User Query cluster
5. Use the following output files:
  - ```BERTopic-cluster-50000-*.json```: contains the keyword information for each User Query cluster
  - ```cluster-info-{n}-{k}.json```: contains the behavior cluster info per each User Query cluster. The raw file should be fixed to match the json sequnece.
  - ```sequences-{n}-{k}.json```: contains the events, search engine performance metrics, and cluster information of each session.\
  - The intermediate files could be kept for re-running the scripts. 