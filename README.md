# [Practice Module] Project Submission

**Archive naming convention:** CourseCode-StartDate-BatchCode-TeamName-ProjectName.zip

This repository follows the [Workshop Project Submission Template](https://github.com/IRS-PM/Workshop-Project-Submission-Template). The application source is under [SystemCode/clips](SystemCode/clips/); the application-specific setup guide is [docs/wearto-runbook.md](SystemCode/clips/docs/wearto-runbook.md).

---

## SECTION 1: PROJECT TITLE

# WearTo — Where to? WearTo!

Intelligent Travel Style & Shopping Assistant · Group 17

---

## SECTION 2: EXECUTIVE SUMMARY / PAPER ABSTRACT

Planning what to wear for a trip means reconciling destination weather, activities, personal style, budget, inspiration, available products and clothes already owned. WearTo brings these inputs into one conversational workflow. The traveller can describe a trip over multiple chat turns, review the interpreted requirements, and refine the resulting recommendations through further feedback.

The current local application uses Qwen for trip understanding, weather and knowledge-based context, Unsplash and FashionCLIP for visual inspiration, and eBay Production image and keyword search for purchasable products. It ranks eBay candidates against the trip context and social references, keeping category Top-K results separate from the user's wardrobe. A compatibility model combines both pools into outfits; the recommendation module applies constraints and presents up to three options. After confirming one option, the user may request an AI outfit preview. Raw intermediate files and generated results are retained locally for inspection and module handoff.

The website is a local single-user prototype. Running the live workflow requires separately configured API credentials, downloaded models and an enabled eBay Production keyset. Product availability reflects API evidence at retrieval time, not a guaranteed checkout outcome.

---

## SECTION 3: CREDITS / PROJECT CONTRIBUTION

| Team member | Student ID | Main contribution |
| --- | --- | --- |
| Agarwal Hritik Kumar | A0353874R | Final recommendation, constraints, feedback loop and outfit preview |
| Jiang Yushan | A0353592X | Trip understanding, dialogue, weather and context reasoning |
| Lu Yujing | A0351286B | Social inspiration retrieval and FashionCLIP reference processing |
| Ma Bofei | A0314662H | Outfit compatibility modelling and combination scoring |
| Zhang Yurun | A0330891A | eBay product retrieval, Top-K ranking, wardrobe understanding and integration |

The [project report](ProjectReport/) will record detailed individual contributions and validation.

---

## SECTION 4: VIDEO OF SYSTEM MODELLING & USE CASE DEMO

The final system presentation and use-case demonstration will be placed in [Video](Video/) for submission.

---

## SECTION 5: USER GUIDE

The complete installation, credential, model, training and website instructions are in the [WearTo runbook](SystemCode/clips/docs/wearto-runbook.md). Start all application commands from **SystemCode/clips**, which contains the Python project and lockfile.

For an already configured checkout:

    cd SystemCode/clips
    .venv/bin/python -m travel_agent.web --port 8765

Open http://127.0.0.1:8765/ in a browser. A fresh checkout first needs the setup steps in the runbook; the repository does not contain private credentials, downloaded model weights or personal wardrobe photos. The [product retrieval runbook](SystemCode/clips/docs/product-retrieval-runbook.md) and [outfit integration guide](SystemCode/clips/docs/outfit-integration.md) explain the module interfaces.

---

## SECTION 6: PROJECT REPORT / PAPER

The final report, including system design, evaluation, limitations and individual reflections, will be placed in [ProjectReport](ProjectReport/) for submission. Current technical documentation and offline evaluation instructions are under [SystemCode/clips/docs](SystemCode/clips/docs/) and [SystemCode/clips/evaluation](SystemCode/clips/evaluation/).

---

## SECTION 7: MISCELLANEOUS

Supplementary submission material, where applicable, will be placed in [Miscellaneous](Miscellaneous/). The application itself, tests, examples and reproducible evaluation code are under [SystemCode/clips](SystemCode/clips/).
