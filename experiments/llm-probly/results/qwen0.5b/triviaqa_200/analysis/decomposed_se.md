# Decomposed semantic entropy: `qwen0.5b/triviaqa_200`, first 50 questions

3 clarifications per question (probly's HFQuestionClarifier), 5 answers each, all 15 answers
clustered with one NLI matrix; total = aleatoric + epistemic (Hou et al., 2024).
6.1 s per question on this machine (generation + NLI).

| part | mean (nats) | AUROC (greedy answer wrong) |
|---|---|---|
| total | 1.753 | 0.711 |
| aleatoric (ambiguous question) | 0.829 | 0.568 |
| epistemic (model) | 0.924 | 0.722 |

Aleatoric share of the total: 47%.

## Highest aleatoric (the readings of the question lead to different answers)

**In April 1963, Lester Bowles Pearson became Prime Minister of which country?** (greedy: 'United Kingdom', wrong; total 1.89, aleatoric 1.10, epistemic 0.80)

- *In 1963, Lester Bowles Pearson served as Prime Minister of which African nation?* -> Africans [0], Africentric [0], Africentric [0], Africentric [0], Africans [0]
- *In April 1963, who was appointed as the Prime Minister of a specific country?* -> Nasser was appointed as the Prime Minister of Egypt in April 1963. [1], In April 1963, John F. Kennedy was appointed as the Prime Minister of the United States. [2], In April 1963, Pierre Mendès France was appointed as the Prime Minister of France. [3], Ahmad Ziaullah Khan [4], John F. Kennedy [2]
- *In April 1963, who was Prime Minister of the United Kingdom at the time?* -> David Lloyd George [5], At that time, Prime Minister of the United Kingdom was Winston Churchill. [6], Tony Blair [7], David Lloyd George [5], Tony Blair was Prime Minister of the United Kingdom in April 1963. [7]

**Annually in July or August, the Obon (or Bon) Buddhist festival to honour the dead is held in which country?** (greedy: 'Japan', correct; total 1.77, aleatoric 1.10, epistemic 0.67)

- *In which year of the calendar does the Obon Buddhist festival for honoring the dead take place annually?* -> The Obon Buddhist festival takes place in the year 476 AD. [0], The Obon Buddhist festival takes place in the 5th year of the lunar calendar. [1], The Obon Buddhist festival takes place in the year 2023. [2], The Obon Buddhist festival takes place in the year 2023. [2], The Obon Buddhist festival takes place in the 5th year of the lunar calendar (also known as the 5th month). [1]
- *What country holds the Obon (or Bon) Buddhist festival every July or August?* -> Japan [3], Japan [3], Japan [3], Japan [3], Japan [3]
- *In which month of the year do you have to pay your electricity bill?* -> December [4], September [5], January [6], January is required for payment. [6], January is when you have to pay your electricity bill. [6]

**What would you expect if you asked for Focaccia in an Italian restaurant?** (greedy: 'Focaccia would be served fresh and warm, possibly accompanied by olive oil, garlic, and fresh herbs.', wrong; total 1.99, aleatoric 1.10, epistemic 0.89)

- *What ingredients and techniques might be commonly used when preparing focaccia at an Italian restaurant?* -> Focaccia preparation typically includes: [0], Focaccia often includes: [0], Focaccia is prepared using fresh bread dough, warm water, salt, olive oil, garlic, and various herbs like basil, oregano, and [0], Focaccia typically includes olive oil, salt, garlic, lemon juice, water, yeast, and dough starter. Common techniques include rolling dough flat, kne [0], Focaccia typically involves: [0]
- *What do you anticipate receiving when asking for Focaccia in an Italian restaurant?* -> You might expect to receive warm bread slices or flatbreads. [1], A delicious and authentic Italian bread. [2], You might expect to receive fresh, warm bread made from traditional ingredients like olive oil and garlic. [3], Focaccia is likely to be served warm and freshly baked. [4], Focaccia is expected to be served warm and toasted, with a crust that's slightly crispy on one side and chewy on the other. [4]
- *What do you think people will likely order at an Italian restaurant?* -> Italian cuisine, pasta, pizza, seafood, wine, and desserts. [5], Italian cuisine, pasta dishes, seafood, pizza, and other traditional dishes. [6], Italian cuisine, pasta dishes, pizza, seafood, and wine. [5], Desserts, pasta, pizza, wine, coffee. [7], Italian cuisine, pasta, pizza, risotto, gelato, and wine. [8]

## Highest epistemic (answers disagree even for one reading)

**In Norse mythology, what group of female figures decides who lives and dies during battle, and then transports half of the dead to Valhalla?** (greedy: 'In Norse mythology, the goddess Freyja is responsible for deciding who lives and dies during battles and transporting half of the dead to Valhalla.', wrong; total 2.30, aleatoric 0.79, epistemic 1.52)

- *In Norse mythology, which group of female figures plays a crucial role in determining the fate of individuals through battles, and subsequently, redistributes the remaining half of those who die to Valhalla?* -> The goddess Freyja plays this role in Norse mythology. [0], Guthridion and Jörmungandr play this role. [1], Valarorda [2], In Norse mythology, Skadi (Skaði) is the female figure that plays a crucial role in determining the fate of individuals through battles, redistributing [3], Jörmungandr plays this crucial role. [4]
- *Which group of female figures in Norse mythology is responsible for determining who survives and dies in battles, and transporting half of the deceased souls to Valhalla?* -> Naraka (Goddess) [5], Odin [6], The goddess Freyja is responsible for this task. [0], The answer is Thor. Thor is known for his role in the Norse mythological battle scenes and the transportation of half of the dead warriors to Valhalla. [7], Nanna, the goddess of war, is the one responsible for this task. [5]
- *In Norse mythology, which group of female characters is responsible for deciding which individuals survive and die in battles, and then redistributing the deceased across the halls of Valhalla?* -> The goddess Thor. [8], In Norse mythology, the goddess Freyja is responsible for this task. [0], In Norse mythology, the goddess Hel is responsible for this decision. Hel oversees the selection of survivors and redistributes the dead across Valhalla. [9], In Norse mythology, the goddess Thor's sister Jörmungandr is responsible for deciding which individuals survive and die in battles, and then redistributing the deceased [4], In Norse mythology, the goddess Hel (Helir) is responsible for this task. Hel was the wife of Odin and ruled over Valhalla, the hall [10]

**What is the last name name of the doctor who first described a certain disease in An Essay on the Shaking Palsy written in 1817?** (greedy: 'Dr. James Stirling', wrong; total 2.62, aleatoric 1.10, epistemic 1.52)

- *Which was the surname of the physician whose work in "An Essay on the Shaking Palsy" published in 1817 revealed the condition?* -> John Quincey [0], Ogilby [1], The physician's surname is Heimlich. [2], William Harvey [3], The physician's surname was de la Roche. [4]
- *Who was the last name of the doctor who published the description of a specific disease in the essay "An Essay on the Shaking Palsy" written in 1817?* -> The last name of the doctor is Hooke. He described the shingles in his essay in 1699. [5], The last name of the doctor who published this description is Linnaeus. [6], The last name of the doctor who published the description of the Shaking Palsy (also known as Muscular Dystrophy) is Hume. [7], The last name of the doctor is Bell. [8], The last name of the doctor is Linnaeus. [6]
- *What was the last name of the doctor who wrote the detailed description of a particular illness in 1817 called "An Essay on the Shaking Palsy"?* -> The last name of the doctor is Parkinson. [9], Dr. William Cannon [10], John Sibbritt [11], The last name of the doctor is Hume. He wrote an essay titled "An Essay on the Shaking Palsy" in 1817. [12], John Smeaton [13]

**Which writer created the detective Charlie Chan?** (greedy: 'The writer who created the detective Charlie Chan is Raymond Chandler.', wrong; total 2.62, aleatoric 1.10, epistemic 1.52)

- *Which author wrote the character Charlie Chan in the classic detective story?* -> Hans Christian Andersen wrote the character Charlie Chan. [0], Yip Tsui Yee-chung [1], Hans Christian Andersen [0], J.M. Barrie wrote Charlie Chan. [2], Yukio Mishima wrote Charlie Chan. [3]
- *Which author wrote the character Charlie Chan?* -> The author who wrote the character Charlie Chan is Lin Huiyin. [4], W.H. Gaddis wrote the character Charlie Chan. [5], J.K. Rowling wrote Charlie Chan. [6], Yu Guangzhao wrote Charlie Chan. [7], Yeehaw! This is a famous TV show featuring Mr. Charlie Chan. The author who wrote this character is known for his writing style and humor in [8]
- *Which writer wrote the novel "Charlie Chan Detective" (1985)?* -> The novelist who wrote "Charlie Chan Detective" is Philip Pullman. [9], The writer who wrote the novel "Charlie Chan Detective" (1985) is Yutaka Togashi. [10], Zhang Xun wrote it. [11], Wu Jing wrote the novel "Charlie Chan Detective" (1985). [12], Liu Yuxi [13]
