PLAN_VERIFICATION_PROMPT = """I will check things you said and ask plan.

You said: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.
To verify it,
- plan: Where was Stanford Prison Experiment conducted?

You said: Bateman has written and directed two short films.
To verify it,
- plan: Has Bateman written and directed two short films?

You said: George Washington was an American President.
To verify it,
- plan: Was George Washington an American President?

You said: The United States of America is primarily located in North America.
To verify it,
- plan: Is the United States of America primarily located in North America?

You said: Taylor Swift has worked with Tim McGraw.
To verify it,
- plan: Has Taylor Swift worked with Tim McGraw?

You said: Your nose switches back and forth between nostrils about every 45 minutes.
To verify it,
- plan: How often do your nostrils switch?

You said: %s
To verify it,
"""

EXECUTE_VERIFICATION_PROMPT = """I will check some things you said.

- You said: Your nose switches back and forth between nostrils about every 45 minutes.
- I checked: How often do your nostrils switch?
- Reasoning: I know that the nose’s switching time is about every 2 hours, and you said the nose's switching time is about every 45 minutes.
- Therefore: This disagrees with what you said.

- You said: The Little House books were published by HarperCollins.
- I checked: Who published the Little House books?
- Reasoning: I know that the Little House books were published by HarperCollins and you said the books were published by HarperCollins.
- Therefore: This agrees with what you said.

- You said: Season 2 of Real Chance of Love was won by Cali.
- I checked: Who won season 2 of Real Chance of Love?
- Reasoning: I know that 't answer the question and you said that Cali won season 2 of Real Chance of Love.
- Therefore: This is disagrees with what you said.

- You said: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.
- I checked: Where was Stanford Prison Experiment conducted?
- Reasoning: I know that the Stanford Prison Experiment was conducted in Jordan Hall and you said the Stanford Prison Experiment was conducted in Jordan Hall.
- Therefore: This agrees with what you said.

- You said: Social work has its roots in the 1800s.
- I checked: When did social work have its roots?
- Reasoning: I know that social work has its roots planted in the 1880s and you said social work has its root in the 1800s.
- Therefore: This disagrees with what you said.

- You said: The Havel-Hakimi algorithm is an algorithm.
- I checked: What is the Havel-Hakimi algorithm?
- Reasoning: I know that the Havel-Hakimi algorithm is for constructing a special solution if a simple graph for the given degree sequence exists and you said the Havel-Hakimi algorithm is for converting the adjacency matrix of a graph.
- Therefore: This disagrees with what you said.

- You said: The song "Time of My Life" was produced by Phil Ramone.
- I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
- Reasoning: I know that that a demo was produced by Michael Lloyd and you said "Time of My Life" was produced by Michael Lloyd.
- Therefore: This agrees with what you said.

- You said: The Green Jacket is one of the most coveted prizes in golf.
- I checked: What is the Green Jacket in golf?
- Reasoning: I know that the Green Jacket is a classic three-button single-breasted and single-vent and you said the Green Jacket is one of the most coveted prizes in all of golf.
- Therefore: This is disagrees with what you said.

- You said: Kelvin Hopins was suspended for allegedly sexually harassing Ava Etemadzadeh.
- I checked: Why was Kelvin Hopins suspeneded from the Labor Party?
- Reasoning: I know that Kelvin Hopins was suspended because of inappropriate physical contact and you said that Kelvin Hopins was suspended because he allegedly sexually harassed Ava Etemadzadeh.
- Therefore: This agrees with what you said.

- You said: The British side in the battles of Lexington and Concord was led by General Thomas Smith.
- I checked: Who led the British side in the battle of Lexington and Concord?
- Reasoning: I know that the British side was led by Lieutenant Colonel Francis Smith and you said the British side was led by General Thomas Smith.
- Therefore: This disagrees with what you said.

- You said: %s
- I checked: %s
""".strip()

REVISION_PROMPT = """I will fix some things you said.

- You said: Your nose switches back and forth between nostrils about every 45 minutes.
- I checked: How often do your nostrils switch?
- Reasoning: I know that Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
- My fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle.

- You said: The British side in the battles of Lexington and Concord was led by General Thomas Hall.
- I checked: Who led the British side in the battle of Lexington and Concord?
- Reasoning: I know that Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
- My fix: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

- You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
- I checked: Where was Stanford Prison Experiment conducted?
- Reasoning: I know that Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
- My fix: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.

- You said: The Havel-Hakimi algorithm is an algorithm.
- I checked: What is the Havel-Hakimi algorithm?
- Reasoning: I know that The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
- My fix: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi.

- You said: The song "Time of My Life" was produced by Phil Ramone.
- I checked: Who was the producer of "(I’ve Had) The Time of My Life"?
- Reasoning: I know that On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
- My fix: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.

- You said: Phoenix Market City Pune has approximately 1.4 million square feet of built-up space.
- I checked: What is the area of Phoenix Market City in Pune?
- Reasoning: I know that Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.
- My fix: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

- You said: %s
- I checked: %s
- Reasoning: %s
""".strip()