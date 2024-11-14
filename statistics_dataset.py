# %%
import datasets

lang = "fr"

query_judgements = datasets.load_dataset(
        "miracl/miracl", 
        lang, 
        # split='train',
        cache_dir="hf_datasets_cache"
    )

# %%
query_judgements

# %%
trains = query_judgements["train"]

# %%
trains

no_of_queries = 0

no_of_positives = 0
no_of_negatives = 0

min_negatives = 40e6
max_negatives = -40e6

min_positives = 40e6
max_positives = -40e6

poss_distr = {i: 0 for i in range(10)}
negs_distr = {i: 0 for i in range(10)}

judge_distr = {i: 0 for i in range(10)}


for row in trains:
    
    id = row["query_id"]
    q_text = row["query"]
    postives = row["positive_passages"]
    negatives = row["negative_passages"]

    no_of_queries += 1
    no_of_positives += len(postives)
    no_of_negatives += len(negatives)


    negs = len(negatives)

    negs_distr[negs] = negs_distr.get(negs, 0) + 1

    if min_negatives > negs:
        min_negatives = negs
    
    if max_negatives < negs:
        max_negatives = negs

    poss = len(postives)

    poss_distr[poss] = poss_distr.get(poss, 0) + 1

    judge_distr[poss+negs] = judge_distr.get(poss+negs, 0) + 1

    if min_positives > poss:
        min_positives = poss
    
    if max_positives < poss:
        max_positives = poss

assert no_of_queries == 1143, f"Expected {no_of_queries} queries but found {no_of_queries}"

print(f"# queries {no_of_queries}")

print()

print(f"avg. # positives {no_of_positives / no_of_queries}")
print(f"min. # positives {min_positives}")
print(f"max. # positives {max_positives}")

print(poss_distr)

print()

print(f"avg. # negatives {no_of_negatives / no_of_queries}")
print(f"min. # negatives {min_negatives}")
print(f"max. # negatives {max_negatives}")

print(negs_distr)

print("====")
print(judge_distr)


