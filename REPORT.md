# Assignment 1 Report

- **Name**: Aitana Damotte Huedo
- **Student ID**: 18654
- **Email**: adamotte.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used the Sephora Products dataset (originally published on Kaggle, scraped from sephora.com in 2023; I downloaded a copy from a public GitHub repo). It's saved in `data/product_info.csv`. Each row is one product sold at Sephora. There are 8,494 rows and 27 columns: brand, price, size, category, number of shades, flags like "Sephora exclusive" or "limited edition", and `loves_count` (how many shoppers saved the product with the heart button).

I picked it because I'm interested in beauty and luxury brands, and because it's messy in interesting ways: sizes are written as text ("3.4 oz/ 100 mL"), and the popularity numbers depend a lot on how long a product has been on the site. That made feature engineering actually matter.

## Business / real-life framing

I imagined I work in Sephora's merchandising team. Every week new products get listed, and we can only give premium promotion (homepage, displays, ads) to some of them. My model answers: **on the day a product is listed, how likely is it to become a hit?** If the probability is above a threshold, we promote it.

This framing drove four decisions:

**1. Target definition.** My first idea was "hit = top 25% of loves". But loves pile up over time: the oldest 10% of products have a median of about 20,000 loves and the newest 10% about 2,900. A raw cutoff would mostly learn "is this product old", which is useless for a product launching today. So I defined a **hit as being in the top 25% of loves among products in the same category listed in the same period** (I split products into 10 equal time periods). This gives a hit rate of 25.3%.

**2. Time split.** There's no launch date column, but Sephora product IDs are handed out in order, so I used the rank of the ID as a "listing clock". Since the model is meant for future products, I split by time: the oldest 70% to train (5,516), the next 15% to validate (1,182) and the newest 15% to test (1,183). A random split would leak the future into training. I checked: with a random split the test AUC would have been 0.853 instead of 0.823, which looks better only because it cheats.

**3. What rows to keep.** I removed the 609 products marked as "new". They've only been listed for a few weeks, so their final popularity isn't known yet. I also removed 4 gift cards (0 loves, not real products). That leaves 7,881 products.

**4. Metric and threshold.** I gave each mistake a cost. Promoting a product that flops (false positive) wastes about $300 of promotion budget. Not promoting a product that becomes a hit (false negative) loses about $1,000 of launch margin. These are my assumptions, and the dashboard lets you change them. Because missing a hit is about 3× worse, the right threshold is well below 0.5: in theory 300 / (300 + 1000) = 0.23 for a perfectly calibrated model. I picked the threshold that minimises total cost on the **validation** set, and only then measured it on test. For comparing models I also report ROC-AUC, PR-AUC (better when only 25% are positives), log-loss and Brier score.

## Data preparation & feature engineering

My main rule: **only use what a buyer knows on listing day.** Rating, number of reviews, loves, out-of-stock status and sale price all happen after launch, so they are never features. Also, everything that "learns" from data (medians, category lists, brand stats, the scaler) is fitted on the training set only, inside a `FeatureBuilder` class in `features.py`. Validation and test rows only go through `.transform()`.

What I built (62 features in total):

- **Price on a log scale**, because prices are very skewed.
- **Relative price**: price compared with the median of its sub-category (is $50 expensive *for a foundation*?), the same for price per mL/g (I parsed the size text into a number), and price compared with the brand's usual price.
- **Brand features**: smoothed brand hit rate, number of products the brand has, and whether the brand is unseen. The hit rate is smoothed toward the overall average (25%) so a brand with only 2 products can't look like a hit factory. For training rows I computed all brand features **out-of-fold** (5 folds): each row's brand stats come from the other 4 folds, so its own label never hides inside its feature. This also means some training rows see an "unknown brand", so the model learns how to handle one.
- **Variants**: number of shades/variants (log) and how much prices vary between them.
- **Flags**: limited edition, online only, Sephora exclusive, has a "value" price (gift sets).
- **Name keywords**: set/kit, mini/travel, refill.
- **Ingredients**: number of ingredients (log), plus a flag when they're missing.
- **One-hot categories**: main category (I merged the tiny "Men" and "Tools & Brushes" into "Other") and sub-categories with at least 30 training products.

Then everything is standardised (mean 0, std 1) so the coefficients are comparable and gradient descent works well.

**Did it help?** I removed one group at a time and measured validation AUC (scikit-learn, same C):

| Version | Validation AUC |
|---|---|
| All features | 0.822 |
| Without brand stats | 0.763 |
| Without categories | 0.802 |
| Without name keywords + ingredients | 0.817 |
| Without relative price | 0.819 |
| Without variants | 0.821 |
| Only raw columns (price, flags, category) | 0.734 |

So the engineering adds about +0.09 AUC, and the brand stats matter most.

**What didn't work: highlight tags.** I built 20 features from the "highlights" tags (Vegan, Cruelty-Free, Clean at Sephora…). Adding them made validation AUC *worse* (0.815). When I investigated, I found **concept drift**: Sephora started tagging products much more over time ("Cruelty-Free" is on 13% of training products but 36% of test products). In the old products, having tags meant a 31% hit rate vs 15% without, probably because popular products were the ones that got re-tagged. In the newest products that pattern disappears (24% vs 32%). The model was learning a pattern that is no longer true, so I turned the tags off (`use_highlights=False`) and kept the code to show the check.

## Modeling: three implementations, one model

My target is binary (hit or not), so I used **logistic regression**. All three versions minimise exactly the same objective: mean binary cross-entropy + an L2 penalty of (1 / (2·C·n))·‖w‖². This is sklearn's penalty rewritten for an average loss, and the intercept is not penalised. 

- **scikit-learn**: `LogisticRegression` with the L-BFGS solver. I tried C in {0.0003 … 1} and chose **C = 0.01** because it had the lowest validation log-loss (0.430).
- **Manual PyTorch**: raw tensors `w` and `b` starting at zero, full-batch gradient descent for 3,000 epochs. Each step: forward pass, loss, `loss.backward()`, update `w -= lr * w.grad` inside `torch.no_grad()`, then zero the gradients. Learning rate 0.5.
- **Standard PyTorch**: an `nn.Module` with one `nn.Linear` layer, a `DataLoader` with mini-batches of 256, the Adam optimizer (lr = 0.01) and a cosine schedule that shrinks the learning rate toward 0 by the end, for 300 epochs.

**How I chose the learning rate:** at first I picked it on validation loss, and the winner was a run that *hadn't finished* converging. Stopping early works like extra regularisation, so validation was rewarding an unfinished model. Since the learning rate is just an optimiser setting (C is the model setting), I now pick it by the training objective reached.

**Test results (1,183 newest products):**

| Model | ROC-AUC | PR-AUC | Log-loss | Brier | Threshold (from val) | Precision | Recall | Cost |
|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.823 | 0.638 | 0.421 | 0.135 | 0.29 | 0.48 | 0.73 | $147,100 |
| Manual PyTorch | 0.823 | 0.638 | 0.421 | 0.135 | 0.29 | 0.48 | 0.73 | $147,100 |
| nn.Module PyTorch | 0.823 | 0.638 | 0.421 | 0.135 | 0.27 | 0.46 | 0.73 | $154,200 |
| Naive baseline | 0.500 | 0.246 | 0.558 | 0.186 | n/a | n/a | n/a | $267,600 |

The naive baseline always predicts the training hit rate (25.3%). Its cost is the cheaper of "promote nothing" ($291,000) and "promote everything" ($267,600). The model saves **$120,500 (45%)**.

**Do they agree?** Yes. The manual loop's probabilities differ from sklearn's by at most 0.0003, and the nn.Module version by at most 0.003. The coefficient correlation is above 0.9999 for both, and all three reach the same training objective (0.40537). This makes sense: logistic regression with L2 is convex, so there's only one minimum and every method that converges finds it. The nn.Module version is a tiny bit further off because mini-batches make each step noisy.

The only visible difference is nn.Module's threshold (0.27 vs 0.29). I investigated: near the minimum the validation cost curve is flat, and 0.27 and 0.29 differ by just $300 (one product). Tiny probability differences tipped it. **At the same threshold (0.29) all three cost exactly $147,100.** So the real lesson is that the exact best threshold is noisy, and anything around 0.25 to 0.31 is about equally good.

The strongest coefficients: number of shades/variants (+0.87) and brand hit rate (+0.85) push toward a hit, while "online only" (−0.34) and Candles & Home Scents (−0.31) push away from it.

## Limitations & next steps

- **New brands.** 22% of test products come from brands not in training, but only 0.4% of training rows look "unseen" even with the out-of-fold trick. Next step: give the model brand information that exists for new brands too (e.g., social-media followers at launch).
- **Listing order is only a proxy.** Product IDs approximate the launch date. With real launch dates I could define time periods more precisely.
- **Loves are not sales.** A heart is interest, not revenue. With sales data I would predict revenue directly.
- **The costs are assumptions.** $300 and $1,000 are my estimates. A real team should replace them with real promotion costs and margins; the dashboard makes that easy.
- **Concept drift.** The tag problem shows the website changes over time. I would retrain regularly and monitor whether feature distributions change.
- **Linear model.** Logistic regression can't capture interactions (e.g., shade count may matter more in makeup than in fragrance). I could add a few interaction features, or later compare with a tree model.

## How to run

```
uv sync
uv run python main.py train   # ~45 s: builds features, trains all three models, saves to models/ and artifacts/
uv run python main.py app     # opens the dashboard; it only loads the saved models
```

Files: `features.py` (cleaning, target, `FeatureBuilder`), `modeling.py` (shared objective, `nn.Module`, loading and metrics), `train.py`, `app.py`.

## Generative AI use disclosure

I used Claude (Anthropic) to help find the dataset, write and debug the code, and draft this report. I reviewed the code and results, and I made sure I can explain every step myself.
