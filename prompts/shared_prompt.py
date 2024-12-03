ATOMIC_TEXT_PROMPT = """I will check and decompose things you said.

You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
I decompose what you said
- I decomposed: Your nose alternates between nostrils.
- I decomposed: During sleep, this alternation happens roughly every 45 minutes.
- I decomposed: The purpose of this alternation is to prevent mucus buildup.
- I decomposed: This process is known as the nasal cycle.

You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
I decompose what you said
- I decomposed: The Stanford Prison Experiment was conducted.
- I decomposed: The location of the experiment was in the basement of Encina Hall.
- I decomposed: Encina Hall is Stanford's psychology building.

You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
I decompose what you said
- I decomposed: The Havel-Hakimi algorithm is a specific method used in graph theory.
- I decomposed: This algorithm is used for converting the adjacency matrix of a graph into its adjacency list.
- I decomposed: The algorithm is named after two individuals, Vaclav Havel and Samih Hakimi.

You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.
I decompose what you said
- I decomposed: "Time of My Life" is a song by American singer-songwriter Bill Medley.
- I decomposed: The song is from the soundtrack of the 1987 film Dirty Dancing.
- I decomposed: The song was produced by Michael Lloyd.

You said: Kelvin Hopins was suspended from the Labor Party due to his membership in the Conservative Party.
I decompose what you said
- I decomposed: Kelvin Hopkins is a member of the Conservative Party.
- I decomposed: Kelvin Hopkins was suspended from the Labor Party due to his membership in the Conservative Party.

You said: Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s.
I decompose what you said
- I decomposed: Social work is a profession.
- I decomposed: This profession is grounded in the philosophical tradition of humanism.
- I decomposed: It is an intellectual discipline.
- I decomposed: The roots of this discipline can be traced back to the 1800s.

You said: %s
I decompose what you said
"""

MERGE_PROMPT = """I will merge some things what I edited. My merge MUST be match some things what I edited and similar to the style of some things what you said.

- You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
- I edited: Your nose switches back and forth between nostrils./ When you sleep, you switch about every 2 hours./ Your nose switches is to prevent a buildup of mucus./ Your nose switches is called the nasal cycle.
- The sentences what I edited said Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle. So, I will merge like this.
- My merge: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle.

- You said: In the battles of Lexington and Concord, the British side was led by General Thomas Hall.
- I edited: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.
- The sentences what I edited said In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith. So, I will merge like this.
- My merge: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

- You said: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
- I edited: The Stanford Prison Experiment was conducted in the basement of Jordan Hall./ Jordan Hall is Stanford’s psychology building.
- The sentences what I edited said The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building. So, I will merge like this.
- My merge: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.

- You said: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
- I edited: The Havel-Hakimi algorithm constructs a special solution./ The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists that one cannot find a positive answer./ The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence proves that one cannot find a positive answer./ The Havel-Hakimi algorithm is named after Vaclav Havel and Samih Hakimi.
- The sentences what I edited said The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi. So, I will merge like this.
- My merge: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi.

- You said: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Phil Ramone.
- I edited: "Time of My Life" is a song by American singer-songwriter Bill Medley./ "Time of My Life" is a song from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd./ "Time of My Life" was produced by Michael Lloyd.
- The sentences what I edited said "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. So, I will merge like this.
- My merge: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.

- You said: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 1.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.
- I edited: Phoenix Market City Pune is located on 21 acres of prime property in Pune./ Phoenix Market City Pune is spread across four levels./ Phoenix Market City Pune is spread across four levels with approximately 3.4 million square feet of built-up space./ Phoenix Market City Pune is owned and operated by Phoenix Mills Limited.
- The sentences what I edited said Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited. So, I will merge like this.
- My merge: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

- You said: The Great Barrier Reef is the world's largest coral reef system composed of over 2,000 individual reefs and 900 islands. It is located off the coast of Queensland, Australia.
- I edited: The Great Barrier Reef is the world's largest coral reef system./ The Great Barrier Reef is composed of over 2,900 individual reefs and 900 islands./ The Great Barrier Reef is located off the coast of Queensland, Australia.
- The sentences I edited said The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands. It is located off the coast of Queensland, Australia. So, I will merge like this.
- My merge: The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands. It is located off the coast of Queensland, Australia.

- You said: Mount Everest is the highest peak in the world, standing at an elevation of 8,848 meters (29,029 feet) above sea level. It is located in the Himalayas, on the border between Nepal and Tibet. Climbing Mount Everest is a challenging and dangerous endeavor, with many climbers attempting to reach its summit each year.
- I edited: Mount Everest, standing at an elevation of 8,848 meters (29,029 feet) above sea level, is the highest peak in the world. It is situated in the Himalayas, straddling the border between Nepal and Tibet. Climbing the mountain is a highly challenging and perilous feat, attracting numerous climbers who aspire to conquer its summit annually.
- The sentences I edited said Mount Everest is the highest peak in the world, standing at an elevation of 8,848 meters (29,029 feet) above sea level. It is located in the Himalayas, on the border between Nepal and Tibet. Climbing Mount Everest is a highly challenging and perilous feat. So, I will merge like this.
- My merge: Mount Everest, standing at an elevation of 8,848 meters (29,029 feet) above sea level, is the highest peak in the world. It is situated in the Himalayas, straddling the border between Nepal and Tibet. Climbing the mountain is a highly challenging and perilous feat.

- You said: %s
- I edited: %s
- The sentences what I edited said 
""".strip()