# Decomposed semantic entropy: `qwen0.5b/triviaqa_200`, first 50 questions

3 clarifications per question (probly's HFQuestionClarifier), 5 answers each, all 15 answers
clustered with one NLI matrix; total = aleatoric + epistemic (Hou et al., 2024).
4.2 s per question on this machine (generation + NLI).

| part | mean (nats) | AUROC (greedy answer wrong) |
|---|---|---|
| total | 1.733 | 0.759 |
| aleatoric (ambiguous question) | 0.802 | 0.597 |
| epistemic (model) | 0.930 | 0.802 |

Aleatoric share of the total: 46%.

## Highest aleatoric (the readings of the question lead to different answers)

**Christina Kirchner was elected as the first female President of which South American country, in 2007?** (greedy: 'Brazil', wrong; total 1.81, aleatoric 1.10, epistemic 0.71)

- *Who was the first woman to serve as the President of Argentina in 2007?* -> Nicolás Maduro [0], The first woman to serve as the President of Argentina in 2007 was Cristina Fernández de Kirchner. [1], María Eugenia Vidal [2], The first woman to serve as the President of Argentina in 2007 was Cristina Fernandez de Kirchner. [1], The first woman to serve as the President of Argentina in 2007 was Cristina Fernandez de Kirchner. [1]
- *Christina Kirchner was appointed as the first female president of what South American nation, in 2007?* -> Brazil [3], Brazil [3], South Africa [4], Brazil [3], South Africa [4]
- *Who was the first female president elected in 2007, and who is she from?* -> The first female president elected in 2007 was Hillary Clinton. She was from New York. [5], The first female president elected in 2007 was Michelle Obama. She is from Illinois, USA. [6], The first female president elected in 2007 was Hillary Clinton. She was from New York. [5], The first female president elected in 2007 was Hillary Clinton, from New York. [5], The first female president elected in 2007 was Hillary Clinton. [5]

**Formula 1 driver Ayrton Senna died while leading the 1994 San Marino Grand Prix at which motor racing circuit?** (greedy: 'Formula 1 Grand Prix of San Marino', wrong; total 2.34, aleatoric 1.10, epistemic 1.24)

- *Formula 1 driver Ayrton Senna died in the final race of the 1994 San Marino Grand Prix, taking advantage of the lead position on lap 63.* -> Ayrton Senna died in the final race of the 1994 San Marino Grand Prix due to the lead position. [0], Ayrton Senna died in the 1994 San Marino Grand Prix. [0], Ayrton Senna died in the final race of the 1994 San Marino Grand Prix due to a pit stop malfunctioning his engine. [1], Ayrton Senna died in 1994's San Marino Grand Prix. [0], Ayrton Senna died in 1994 Grand Prix due to leading position. [0]
- *Which motor race did Formula 1 driver Ayrton Senna die in during his championship-winning year (1994) while leading the race?* -> Formula 1 driver Ayrton Senna died in the Monaco Grand Prix during his championship-winning year (1994). [2], Formula One driver Ayrton Senna died in a motorcycle accident during his championship-winning year (1994). [3], Ayrton Senna died in the Spanish Grand Prix of 1994 while leading the race. [4], Ayrton Senna died in a motorcycle accident during the 1994 Monaco Grand Prix. [5], Ayrton Senna died in the Belgian Grand Prix of 1994 while leading the race. [6]
- *Formula 1 driver Ayrton Senna died in a motorcycle accident during the 1994 San Marino Grand Prix at which motor racing circuit?* -> Formula 1 Grand Prix Circuit of Monza [7], Formula 1 Grand Prix Circuit of the South of France [8], Formula 1 Grand Prix Circuit, also known as the Autodromo Enzo e Dino Ferrari in Italy, where Ayrton Senna was involved in [9], Formula 1 Grand Prix赛道 [10], Formula 1 Grand Prix Circuit San Marino [11]

**'Professor Cuthbert Calculus' is an associate of which character?** (greedy: 'Professor Cuthbert Calculus is an associate of Professor Cuthbert.', wrong; total 2.17, aleatoric 1.10, epistemic 1.07)

- *What is Professor Cuthbert Calculus' relationship with any specific character?* -> Professor Cuthbert Calculus does not have a specific character relationship. [0], Professor Cuthbert Calculus has no direct relationship with any specific character in the provided information. [0], Professor Cuthbert Calculus has no specific relationships with any characters in the story. [0], Professor Cuthbert Calculus has no direct relationship with any specific character in "The Three-Body Problem". [0], Professor Cuthbert Calculus does not have any specific character relationships in this scenario. [0]
- *Which character is Professor Cuthbert Calculus associated with?* -> Professor Cuthbert Calculus is often associated with Dr. William Halsey, an anthropologist and historian. [1], Professor Cuthbert Calculus is associated with Dr. Benjamin Cuthbert. [2], Professor Cuthbert Calculus is often associated with Dr. Peter Cuthbert. [3], Professor Cuthbert Calculus is associated with Sir Isaac Newton. [4], Professor Cuthbert Calculus is associated with Dr. Michael Jordan. [5]
- *What character does Professor Cuthbert Calculus work as an associate of? * -> Professor Cuthbert Calculus works as an associate of Dr. Emily Green. [6], Professor Cuthbert Calculus works as an associate of Dr. James Watson. [7], Professor Cuthbert Calculus works as an associate of Dr. Elizabeth Simmons. [8], Professor Cuthbert Calculus works as an associate of Dr. Jameson, a renowned neuroscientist. [9], Professor Cuthbert Calculus works as an associate of Dr. Emily Smith. [10]

## Highest epistemic (answers disagree even for one reading)

**New Zealander Jack Lovelock gained fame in the1930's in which field?** (greedy: 'New Zealand rugby', wrong; total 2.62, aleatoric 1.01, epistemic 1.61)

- *What was Jack Lovelock known for during his New Zealand years?* -> Jack Lovelock is best known for his work on solar energy and wind power in New Zealand. [0], Jack Lovelock was known for his work in radio astronomy and astrophysics during his New Zealand years. [1], Jack Lovelock was known for developing the first nuclear power plant in New Zealand. [2], Jack Lovelock is best known for his pioneering work in biofuels and biorefineries. [3], Jack Lovelock was known for inventing and developing the first modern gas turbine engine in New Zealand. [4]
- *Which field did New Zealand born Jack Lovelock become famous in after his arrival in 1930?* -> Jack Lovelock became famous for inventing the first practical radio telescope in 1930. [5], Jack Lovelock became famous for his work on solar energy and biomimicry. [6], Jack Lovelock became famous for inventing the first modern solar water heater in 1930. [7], Jack Lovelock became famous for his research on solar energy and photovoltaic technology after arriving in New Zealand in 1930. [8], Jack Lovelock became famous for his work on solar energy and renewable power systems. [0]
- *Who was the famous New Zealander Jack Lovelock who achieved prominence in the 1930s?* -> Jack Lovelock, known for his pioneering work on radioactivity and radiation protection, gained international acclaim during the 1930s. [9], Jack Lovelock, known as "The Father of the Modern World," was a renowned New Zealander who made significant contributions to modern physics and technology. [10], Jack Lovelock, known for his groundbreaking work on radar technology, became well-known in the 1930s. [11], Jack Lovelock became prominent in the 1930s through his work in radio broadcasting and television. [12], Jack Lovelock was an accomplished New Zealander known for his innovative designs and contributions to aviation during the 1930s. [13]

**Which writer created the detective Charlie Chan?** (greedy: 'The writer who created the detective Charlie Chan is Raymond Chandler.', wrong; total 2.62, aleatoric 1.01, epistemic 1.61)

- *Who wrote the detective character Charlie Chan?* -> The detective character Charlie Chan is written by Jack London. [0], Charlie Chan was written by Donald A. Wollheim. [1], The detective character Charlie Chan was written by Donald Judd. [2], Charlie Chan was written by Lin Shuhao and Wu Jing. [3], The detective character Charlie Chan was written by Arthur Conan Doyle. [4]
- *Who is the author of the novel "Detective Charlie Chan"?* -> The author of the novel "Detective Charlie Chan" is Wu Yifan. [5], The author of the novel "Detective Charlie Chan" is Yeh Shuhao. [6], The author of the novel "Detective Charlie Chan" is J.R.R. Tolkien. [7], Wang Mengqing [8], J.D. Salinger [9]
- *Which author wrote the novel "Charlie Chan"?* -> The author who wrote "Charlie Chan" is Yip Tung-kwong. [10], J.M. Barrie wrote "Charlie Chan". [11], Wang Mengyu wrote the novel "Charlie Chan". [12], J.D. Salinger [9], Yuan Guoqian wrote the novel "Charlie Chan". [13]

**Which Frenchman, a football administrator and head of FIFA from 1921 to1954 is regarded as the founder of the football World Cup?** (greedy: 'Paul Pogba', wrong; total 2.62, aleatoric 1.01, epistemic 1.61)

- *Which Frenchman was an administrator and head of FIFA from 1921 to 1954 who established the world cup as an international sporting competition?* -> Lionel Messi [0], Charles Guesne de Bovary [1], Georges Bortolosi [2], The answer is François Chateaubriand. [3], Pierre de Coubertin [4]
- *Which Frenchman, an administrator of the Football Federation who served as head of FIFA from 1921 to 1954, is considered to have founded the World Cup football competition?* -> The Frenchman considered to be the founder of the World Cup was Lionel Messi. [0], Paul Guesde [5], Gian Carlo Gennaro [6], Georges Seville [7], Michel Platini [8]
- *Which former French football administrator who served as the president of FIFA from 1921 to 1954 has been officially recognized as the person who founded the World Cup?* -> Paul Gervais [9], Paul Delahaye [10], Guillermo Sáenz de León [11], Jean-Baptiste Jules de Menon. [12], Guillaume de Gaulle [13]

