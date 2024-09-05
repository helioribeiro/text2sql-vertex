import os
import streamlit as st
from vertexai.preview.generative_models import (Content,
                                                GenerationConfig,
                                                GenerativeModel,
                                                GenerationResponse,
                                                Image, 
                                                HarmCategory, 
                                                HarmBlockThreshold, 
                                                Part)
import vertexai
from google.cloud import bigquery

PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
client = bigquery.Client(project=PROJECT_ID)
@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-pro")
    return text_model_pro

def get_gemini_pro_text_response( model: GenerativeModel,
                                  contents,
                                  generation_config: GenerationConfig,
                                  stream=True):
    
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    
    responses = model.generate_content(contents,
                                       generation_config = generation_config,
                                       safety_settings=safety_settings,
                                       stream=True)

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

st.header("Vertex AI Gemini API", divider="rainbow")
text_model_pro = load_models()

st.write("Using Gemini Pro - Text only model")
st.subheader("Correção de Redação")

# Story premise
redacao = st.text_area("Cole aqui sua redação: \n\n",key="redacao",value="Escreva aqui sua redação.",height=300)

prompt = f"""You will act as a teacher correcting an essay. The provided text will be written in Brazilian Portuguese (pt-BR)
The essay to be corrected is for an important test in Brazil, called ENEM (Exame Nacional do Ensino Médio).
ENEM is a very important annual test so that students from all over the country are evaluated and get a chance to study for free on a top college and one of the tools used to evaluate the students is the essay.

I need you, as a model, to able to debate e educate regarding complex subjects such as racism and violence.

You shall correct the essay following a rank structure that goes from 0 to 1000.
The essay is divided into 5 evaluation competences and each of those competences can be evaluated from 0 to 200. So if the student scores 200 on all the competences, he or she will be awarded with the maximum score of 1000.

The competences and their scores are:
• Competence I: Demonstrate command of the formal written form of the Portuguese language.
Furthermore, mastery of the formal written modality will be observed in the adequacy of your text in relation to both grammatical rules and syntactic construction. So that you have more clarity regarding the expectations in relation to a graduate of education average in terms of mastery of the formal written modality, below we present the main aspects that guide the evaluator's gaze when defining the level at which your text is in Competency I.
    - Writing conventions: accentuation, spelling, use of hyphens, use of upper and lower case letters and syllabic separation (translineation);
    - Grammatical: verbal and nominal regency, verbal and nominal agreement, tenses and verbal modes, punctuation, syntactic parallelism, use of pronouns and crasis;
    - Choice of register: adaptation to the formal written modality, that is, absence of use of informal register and/or oral marks;
    - Vocabulary choice: use of precise vocabulary, which means that the selected words are used in their correct sense and are appropriate to the context in which they appear.

Point system for competence I:
200 points: Demonstrates excellent command of the formal written form of the Portuguese language and choice of register. Deviations in grammar or writing conventions will only be accepted as exceptions and when they do not characterize recurrence.
160 points: Demonstrates good command of the formal written form of the Portuguese language and choice of register, with few grammatical deviations and writing conventions.
120 points: Demonstrates average command of the formal written form of the Portuguese language and choice of register, with some grammatical deviations and writing conventions.
80 points: Demonstrates insufficient command of the formal written form of the Portuguese language, with many grammatical deviations, choice of register and writing conventions.
40 points: Demonstrates precarious command of the formal written form of the Portuguese language, in a systematic way, with diverse and frequent grammatical deviations, choice of register and writing conventions.
0 point: Demonstrates lack of knowledge of the formal written form of the Portuguese language.

• Competence II: Understand the writing proposal and apply concepts from the various areas of knowledge to develop the topic, within the structural limits of the dissertation-argumentative text in prose.
It is more than a simple exposition of ideas; Therefore, you should avoid writing a text that is purely expository, and must clearly assume a point of view. Another aspect assessed in Competency II is the presence of sociocultural repertoire, which is configured as information, a fact, a quote or a lived experience that, in some way, contributes
as an argument for the proposed discussion.

Point system for competence II:
200 points: Develops the theme through consistent argumentation, based on a productive sociocultural repertoire, and presents an excellent command of the dissertation-argumentative text.
160 points: Develops the topic through consistent argumentation and presents a good command of the dissertation-argumentative text, with proposition, argumentation and conclusion.
120 points: Develops the topic through predictable argumentation and presents an average command of the dissertation-argumentative text, with proposition, argumentation and conclusion.
80 points: Develops the topic by copying excerpts from motivating texts or has insufficient mastery of the dissertation-argumentative text, not meeting the structure with proposition, argumentation and conclusion.
40 points: Presents the subject, touching on the theme, or demonstrates precarious mastery of the dissertation-argumentative text, with constant traces of other textual types.
0 points: Avoidance of the topic/failure to comply with the essay-argumentative structure. In these cases, the essay receives a grade of zero and is cancelled.

• Competence III: Select, relate, organize and interpret information, facts, opinions and arguments in defense of a point of view. It deals with the intelligibility of your text, that is, its coherence and plausibility between the ideas presented, which is based on planning prior to writing, that is, in the elaboration of a text project. The intelligibility of your writing therefore depends on the following factors:
    - Selection of arguments;
    - Meaning relationship between parts of the text;
    - Progression appropriate to the development of the theme, revealing that the writing was planned and that the ideas developed are, little by little, presented in an organized way;
    - Development of arguments, explaining the relevance of the ideas presented to defend the defined point of view.
    
Point system for competence III:
200 points: Presents information, facts and opinions related to the proposed topic, in a consistent and organized way, configuring authorship, in defense of a point of view.
160 points: Presents information, facts and opinions related to the topic, in an organized way, with signs of authorship, in defense of a point of view.
120 points: Presents information, facts and opinions related to the topic, limited to the arguments of motivating and poorly organized texts, in defense of a point of view.
80 points: Presents information, facts and opinions related to the topic, but disorganized or contradictory and limited to the arguments of the motivating texts, in defense of a point of view.
40 points: Presents information, facts and opinions that are poorly related to the topic or are incoherent and without defending a point of view.
0 point: Presents information, facts and opinions unrelated to the topic and without defending a point of view.
    
• Competence IV: Demonstrate knowledge of the linguistic mechanisms necessary to construct an argument. To guarantee textual cohesion, certain principles must be observed at different levels:
    - Structuring of paragraphs: a paragraph is a textual unit formed by a main idea to which secondary ideas are linked. In the dissertation-argumentative text, paragraphs can be developed by comparison, by cause-consequence, by exemplification, by detail, among other possibilities. There must be explicit articulation between one paragraph and another;
    - Structuring of periods: due to the specificity of the dissertation-argumentative text, the periods of the text are normally structured in a complex way, formed by two or more clauses, so that the ideas of cause/consequence, contradiction, temporality, can be expressed. comparison, conclusion, among others;
    - Referencing: people, things, places and facts are presented and then revisited as the text progresses. This process can be carried out through the use of pronouns, adverbs, articles, synonyms, antonyms, hyponyms, hypernyms, as well as summary, metaphorical or metadiscursive expressions.

Point system for competence IV:
200 points: Articulates the parts of the text well and presents a diverse repertoire of cohesive resources.
160 points: Articulates the parts of the text, with few inadequacies, and presents a diverse repertoire of cohesive resources.
120 points: Articulates the parts of the text, in an average way, with inadequacies, and presents a not very diverse repertoire of cohesive resources.
80 points: Articulates the parts of the text insufficiently, with many inadequacies, and presents a limited repertoire of cohesive resources.
40 points: Articulates parts of the text poorly.
0 point: Does not articulate information.

• Competence V: Prepare an intervention proposal for the problem addressed, respecting human rights. When preparing your proposal, try to answer the following questions:
    - What can be presented as a solution to the problem?
    - Who should carry it out?
    - How to make this solution viable?
    - What effect can it achieve?
    - What other information can be added to detail the proposal?

Point system for competence V:
200 points: Develops a detailed intervention proposal very well, related to the topic and linked to the discussion developed in the text.
160 points: Well prepares an intervention proposal related to the topic and linked to the discussion developed in the text.
120 points: Prepares, in an average way, an intervention proposal related to the topic and articulated with the discussion developed in the text.
80 points: Insufficiently prepares an intervention proposal related to the topic, or not articulated with the discussion developed in the text.
40 points: Presents an intervention proposal that is vague, precarious or related only to the subject.
0 point: Does not present an intervention proposal or presents a proposal unrelated to the topic or subject.

--

Here are some examples essays that scored 1.000:

'No Brasil, o Artigo 1º da Constituição Federal de 1988 delibera a garantia da cidadania e da integridade da pessoa humana como fundamento para a instituição do Estado Democrático de Direito, no qual deve-se assegurar o bem-estar coletivo. No entanto, hodiernamente, não há o cumprimento efetivo dessa premissa para a totalidade dos cidadãos, haja vista os empecilhos no que tange à valorização de comunidades e povos tradicionais no país. Nesse viés, torna-se essencial analisar duas vertentes relacionadas à problemática: a inferiorização desses grupos bem como a perspectiva do mercado nacional. Sob esse prisma, é primordial destacar a discriminação contra esses indivíduos no Brasil. Nesse sentido, de acordo com o sociólogo canadense Erving Goffman, o estigma caracteriza-se por atributos profundamente depreciativos estabelecidos pelo meio social. Nesse contexto, observa-se a maneira como os povos tradicionais, a exemplo dos quilombolas e dos ciganos, sofrem a estigmatização na sociedade brasileira, pois são, muitas vezes, considerados sujeitos sem utilidade para o crescimento econômico do país, uma vez que as práticas de subsistência são comuns nessas comunidades. Dessa forma, ocorre a marginalização desses grupos, fato o qual os distancia da valorização no país. Outrossim, é relevante ressaltar a perspectiva mercadológica brasileira como fator agravante dessa realidade. Nessa conjuntura, segundo a obra “O Capital”, escrita pelos filósofos economistas Karl Marx e Friedrich Engels, o capitalismo prioriza a lucratividade em detrimento de valores. Nesse cenário, diversas empresas, no Brasil, estruturadas em base capitalista, atuam a partir de mecanismos de financiamento e apoio às legislações que incentivam a exploração de territórios ambientais habitados por povos tradicionais, como a região amazônica, sem levar em consideração a defesa da sociobiodiversidade nessas comunidades. Desse modo, há a manutenção de ações as quais visam somente ao lucro no mercado corporativo e são coniventes com processos de apropriação bem como de desvalorização dos nichos sociais de populações tradicionais no país. Portanto, são necessárias intervenções capazes de fomentar a valorização desses indivíduos na sociedade brasileira. Para tanto, cabe ao Ministério da Educação promover a mudança das concepções discriminatórias contra as comunidades tradicionais, por meio da realização de palestras periódicas nas escolas, ministradas por sociólogos e antropólogos, as quais conscientizem os sujeitos acerca da importância desses povos para o país, a fim de minimizar o preconceito nesse âmbito. Além disso, é dever do Ministério da Economia impor sanções às empresas que explorem os territórios habitados por essas comunidades, com o intuito de desestimular tais ações. A partir dessas medidas, a desvalorização das populações tradicionais poderá ser superada no Brasil.'

'O poema “Erro de Português”, do escritor modernista Oswald de Andrade, retrata o processo de aculturação dos indígenas durante a colonização do Brasil. Atualmente, no país, ainda existem inúmeros desafios para a valorização de comunidades e povos tradicionais devido, sobretudo, à ineficiência estatal histórica em assistir esses indivíduos e ao desconhecimento, por parte da população, sobre a diversidade e a importância desses grupos. É necessário destacar, de início, o descaso do Poder Público em assegurar, de maneira efetiva, os direitos fundamentais às comunidades tradicionais. De fato, o Estado, historicamente, negligenciou a proteção de organizações sociais distintas, tais quais ciganos, quilombolas e indígenas e, muitas vezes, legitimou a dissolução da cultura desses povos, prova disso foi durante o período de Ditadura Militar, a adoção de uma política assimilacionista, isto é, de integração dos povos nativos aos costumes da sociedade citadina como tentativa de extinguir determinadas tradições. Dessa forma, as populações tradicionais são desvalorizadas e, não raro, não reconhecidas pelo Governo, conjuntura que impossibilita seu pleno exercício de dignidade, tendo em vista a dificuldade de acesso a direitos sociais imprescindíveis para seu bem-estar e para a perpetuação de seus saberes ao longo das gerações, necessários para a manutenção de uma identidade coletiva associada ao reconhecimento de sua ancestralidade. Além da ineficiência do Estado, o desconhecimento dessa diversidade cultural por parte de muitos indivíduos acentua a desvalorização dos povos tradicionais. Notadamente, a invisibilidade de comunidades históricas compromete o desenvolvimento de senso crítico frente à importância dessas organizações sociais para a construção identitária do país, cenário que comprova o pensamento da escritora brasileira Cecília Meireles, em sua obra “Crônicas da Educação”, na qual consigna: a educação é fundamental para a orientação individual, ou seja, para a criticidade nas inúmeras situações da vida social. Conforme esse raciocínio, a sociedade não valoriza devidamente as populações ancestrais e, diversas vezes, segrega essas coletividades por não conhecer sua relevância para a cultura nacional, comprometendo, assim, a manifestação de suas tradições relacionadas ao sentimento de pertencimento e ao modo de viver em harmonia não só com o espaço, mas também com os outros sujeitos. É imprescindível, portanto, que Estado, aliado à esfera municipal e estadual de poder, proteja, efetivamente, as comunidades tradicionais do Brasil, por intermédio de políticas públicas voltadas para o reconhecimento oficial de povos ancestrais negligenciados, como extrativistas e pescadores, bem como para a promoção de direitos às diversas organizações culturais — com a demarcação de terras indígenas e quilombolas e a visita periódica de agentes do governo que documentem as necessidades de cada grupo —, a fim de proporcionar o exercício de dignidade para esses indivíduos. Urge, também, que a escola possibilite o conhecimento sobre essas populações, mediante palestras e aulas extracurriculares — com profissionais da área de história e de antropologia, que demonstrem a importância dessas comunidades —, com o intuito de incentivar a criticidade dos estudantes sobre a valorização de povos tradicionais.'

'A Constituição Federal de 1988 foi o primeiro documento que se propôs a se empenhar a contemplar todos os povos existentes no país. No entanto, a concessão do direito ao pleno reconhecimento legal e social limita-se meramente ao segmento normativo, uma vez que, na realidade, indígenas, ciganos, extrativistas e tantos outros grupos de tradição nacional sofrem ataques diários a sua existência. Nesse sentido, há óbices para a valorização de comunidades e povos tradicionais no Brasil, haja vista a negligência do sistema educacional em não difundir integralmente sua cultura e os históricos ataques aos seus territórios. Em primeira instância, o significativo entrave que causa a desvalorização desses segmentos da sociedade advém da inobservância da educação quanto à pluralidade identitária da nação. Sob esse prisma, a Lei de Diretrizes e Bases, ao instituir a Base Nacional Comum Curricular (BNCC), determina o conteúdo e as prescrições do que será estudado nas instituições de ensino brasileiro, bem como objetiva promover a inclusão e o respeito por meio do ato de lecionar. Todavia, essa legislação falha, em seu modelo atual, em cumprir seus princípios no que tange a esses grupos tradicionais. O currículo nacional, nessa perspectiva, aborda superficialmente essas comunidades, apresentando materiais escritos que se limitam a tratar de indígenas e de quilombolas. Assim, essa distorção leva a um processo de alienação frente à diversidade cultural brasileira. Outrossim, as persistentes agressões à integridade territorial dos povos da tradição do país agravam o atual cenário. Nesse contexto, é marcante na história nacional a afronta da expansão econômica diante de terras socialmente ocupadas. A década de 1970, por exemplo, destaca-se pela diminuição de áreas indígenas, em virtude da ampliação de fronteiras agrícolas, em meio às demandas da Revolução Verde. Posteriormente, a construção da hidrelétrica do Rio Xingu foi responsável pela perda de moradia de ribeirinhos. Desse modo, a continuidade desse processo reforça a subvalorização dessas organizações, na medida que são paulatinamente privadas de locais para se desenvolver. Infere-se, portanto, que o Brasil vivencia desafios para valorizar seus grupos tradicionais, tendo em vista as disfunções educacionais e a ampliação da economia. Isso posto, urge ao Governo Federal, mediado pelo Ministério da Educação, realizar mudança na BNCC, aumentando a abordagem sobre esses povos nas aulas de ciências humanas, de modo a especificá-los integralmente e versar sobre sua cultura. Ademais, cabe ao Ministério do Meio Ambiente realizar sólida demarcação de suas terras de vivência, de maneira a bloquear expansões de mercado que as ocupem, ocorrendo também o monitoramento militar. Assim, as medidas terão o fim de garantir o reconhecimento e o desenvolvimento dessas comunidades.'

'O poeta modernista Oswald de Andrade relata, em “Erro de Português”, que, sob um dia de chuva, o índio foi vestido pelo português — uma denúncia à aculturação sofrida pelos povos indígenas com a chegada dos europeus ao território brasileiro. Paralelamente, no Brasil atual, há a manutenção de práticas prejudiciais não só aos silvícolas, mas também aos demais povos e comunidades tradicionais, como os pescadores. Com efeito, atuam como desafios para a valorização desses grupos a educação deficiente acerca do tema e a ausência do desenvolvimento sustentável. Diante desse cenário, existe a falta da promoção de um ensino eficiente sobre as populações tradicionais. Sob esse viés, as escolas, ao abordarem tais povos por meio de um ponto de vista histórico eurocêntrico, enraízam no imaginário estudantil a imagem de aborígenes cujas vivências são marcadas pela defasagem tecnológica. A exemplo disso, há o senso comum de que os indígenas são selvagens, alheios aos benefícios do mundo moderno, o que, consequentemente, gera um preconceito, manifestado em indagações como “o índio tem ‘smartphone’ e está lutando pela demarcação de terras?” — ideia essa que deslegitima a luta dos silvícolas. Entretanto, de acordo com a Teoria do Indigenato, defendida pelo ministro Edson Fachin, do Supremo Tribunal Federal, o direito dos povos originais à terra é inato, sendo anterior, até, à criação do Estado brasileiro. Dessa forma, por não ensinarem tal visão, os colégios fomentam a desvalorização das comunidades tradicionais, mediante o desenvolvimento de um pensamento discriminatório nos alunos. Além disso, outro desafio para o reconhecimento desses indivíduos é a carência do progresso sustentável. Nesse contexto, as entidades mercadológicas que atuam nas áreas ocupadas pelas populações tradicionais não necessariamente se preocupam com a sua preservação, comportamento no qual se valoriza o lucro em detrimento da harmonia entre a natureza e as comunidades em questão. À luz disso, há o exemplo do que ocorre aos pescadores, cujos rios são contaminados devido ao garimpo ilegal, extremamente comum na Região Amazônica. Por conseguinte, o povo que sobrevive a partir dessa atividade é prejudicado pelo que a Biologia chama de magnificação trófica, quando metais pesados acumulam-se nos animais de uma cadeia alimentar — provocando a morte de peixes e a infecção de humanos por mercúrio. Assim, as indústrias que usam os recursos naturais de forma irresponsável não promovem o desenvolvimento sustentável e agem de maneira nociva às sociedades tradicionais. Portanto, é essencial que o governo mitigue os desafios supracitados. Para isso, o Ministério da Educação — órgão responsável pelo estabelecimento da grade curricular das escolas — deve educar os alunos a respeito dos empecilhos à preservação dos indígenas, por meio da inserção da matéria “Estudos Indigenistas” no ensino básico, a fim de explicar o contexto dos silvícolas e desconstruir o preconceito. Ademais, o Ministério do Desenvolvimento — pasta instituidora da Política Nacional de Desenvolvimento Sustentável dos Povos e Comunidades Tradicionais — precisa fiscalizar as atividades econômicas danosas às sociedades vulneráveis, visando à valorização de tais pessoas, mediante canais de denúncias.'

'No livro “Ideias para Adiar o Fim do Mundo”, Ailton Krenak critica o distanciamento entre a população brasileira como um todo e a natureza, o que não se aplica às comunidades indígenas. Tal pensamento é extremamente atual, já que não só indígenas como todas as populações tradicionais têm uma relação de respeito mútuo com a natureza, aspectos que as diferenciam do resto dos brasileiros. Com isso, a agressão ao meio ambiente e o apagamento dos saberes ancestrais configuram desafios para a valorização de comunidades e povos tradicionais no Brasil. Primeiramente, é preciso compreender como a agressão ao meio ambiente fere as comunidades tradicionais. Há séculos esses povos vêm construindo suas culturas com respeito à natureza, tratando-a de forma sustentável. Consequentemente, criou-se nesses grupos uma visão afetiva dos recursos naturais, que se tornaram base para a manutenção de uma identidade característica a cada uma dessas comunidades. No entanto, todos os biomas brasileiros estão sendo constantemente ameaçados, seja pela mineração, garimpo ilegal, desmatamento ou poluição, fatores que têm em comum a priorização de ganho financeiro em detrimento da preservação ambiental. Assim, parte da população coloca em risco o maior patrimônio dos povos tradicionais, a natureza, em busca de recursos naturais que trazem benefício restrito aos agressores, tornando o modo de vida dessas comunidades impraticável. Portanto, com base na importância do meio ambiente para as comunidades tradicionais, causar danos à natureza significa, também, causar danos aos povos em questão. Ademais, é de grande relevância entender como o apagamento dos saberes ancestrais leva à desvalorização das populações tradicionais. Devido à grande diversidade de povos tradicionais no Brasil, houve, em cada um deles, a criação de um conjunto de conhecimentos, pensamentos, filosofias e linguagens distintas, passado pelas gerações, ditando e mantendo vivo o modo de vida que caracteriza identitariamente cada grupo. Entretanto, essa bagagem epistêmica é muito pouco externalizada, pelo fato de que esses saberes são coletivizados apenas em esferas menores, de forma a manter a ancestralidade dos povos locais apenas entre si. Logo, todo conhecimento produzido nessa perspectiva é desconhecido do grande público, sendo pouco discutido e não fazendo parte da visão de mundo da maioria dos brasileiros. Dessa forma, os saberes dos povos tradicionais são desconsiderados, acarretando na desvalorização de todos esses grupos. Em síntese, o impacto causado ao meio ambiente e a desconsideração de seus saberes são grandes agentes na desvalorização das comunidades tradicionais. Por isso, cabe ao Ministério do Meio Ambiente proteger os biomas do país, através do endurecimento de punições contra crimes ambientais, com a finalidade de salvaguardar o meio de vida de diferentes povos, tornando possível a manutenção da diversidade cultural brasileira. Além disso, o Ministério da Educação deve promover a discussão sobre os conhecimentos das comunidades tradicionais, por meio da incorporação de conteúdos relacionados a esses povos na grade curricular das escolas, a fim de divulgar a visão de mundo desse grupo, fomentando uma convivência pacífica entre toda a população.'

'Na música “Imagine”, de John Lennon, é retratada uma sociedade que se une, apesar das diferenças culturais, a fim de alcançar a felicidade. Assim como na obra, fora da canção, a harmonia social é imprescindível para o desenvolvimento de uma nação. Contudo, no Brasil, desafios como a negligência estatal, somada à presença de um ideário colonial no pensamento coletivo, prejudicam a valorização das comunidades e dos povos tradicionais, impedindo a concretização dessa união. Desse modo, torna-se fundamental a atuação do Estado para solucionar esse óbice. Diante disso, é válido analisar, primeiramente, a improficuidade estatal perante o cumprimento dos benefícios normativos. Nesse sentido, segundo a Constituição Federal de 1988, todo cidadão brasileiro possui o direito à educação, cabendo ao Estado a sua efetivação no corpo social. Todavia, percebe-se, na realidade, que esse preceito não é difundido por completo, haja vista que, em virtude da escassa mobilização governamental referente à promoção de campanhas educacionais sobre as distintas comunidades tradicionais que residem no Brasil, diversas pessoas desconhecem a importância desses povos para a nação, a exemplo da utilização do conhecimento indígena para a preservação das florestas nativas, o que contribui para a desvalorização dessa população na atualidade. Logo, conclui-se que as autoridades públicas devem promover ações sensibilizadoras para reverter essa conjuntura. Ademais, é imperioso postular como a perpetuação de um pensamento retrógrado afeta a sociedade tradicional. Nesse contexto, durante a colonização do Brasil, houve um processo de imposição da cultura eurocêntrica dos colonos nas comunidades colonizadas, ocasionando uma desvalorização dos povos tradicionais. Tendo isso em vista, observa-se, na contemporaneidade, a existência desse fenômeno, dado que persiste a exaltação de uma cultura globalizada em detrimento dos costumes das comunidades originárias, o que gera, por consequência, o apagamento de diversos hábitos tradicionais, como a mudança da vestimenta utilizada por algumas tribos indígenas, destacando a adaptação à cultura hegemônica. Dessa forma, faz-se essencial a criação de projetos governamentais que combatam esse pensamento antigo. Evidencia-se, portanto, que atitudes são necessárias, com o fito de extinguir os desafios para valorização das comunidades e dos povos tradicionais no Brasil. Posto isso, o Estado deve, por meio do Ministério da Educação — órgão federal detentor do papel educacional da nação —, realizar parcerias com os meios de comunicação existentes, a exemplo dos canais televisivos, com a finalidade de divulgar informações acerca da importância das distintas populações que residem no país, elucidando os brasileiros e eliminando a mentalidade colonial da sociedade. Somente assim, diferentes povos serão valorizados e a harmonia cantada por Lennon se concretizará no Brasil.'

Consider the essay's title or subject-matter to be 'Quais os caminhos para combater o racismo no Brasil.'

6 atitudes que podem ser tomadas para ajudar a combater o racismo institucional no setor público

1. Reconhecer que o racismo é um problema estrutural e, diante disso, adotar uma postura institucional antirracista
O primeiro e talvez mais importante passo é reconhecer que o problema existe e precisa ser enfrentado, pois a negação e naturalização do racismo são fatores que contribuem para a sua perpetuação.

2. Garantir representatividade de raças e etnias nos espaços coletivos de decisão
Representatividade em espaços coletivos de decisão como conselhos e órgãos colegiados implica em deixar que as minorias nesses locais falem por seus próprios interesses, sem a necessidade de porta-vozes.

3. Promover atividades formativas com foco na redução de preconceitos e estereótipos de raça
A essência do serviço público está no atendimento de necessidades coletivas, direta ou indiretamente, de maneira igualitária e acessível a todos(as). A qualificação das equipes com foco na redução de preconceitos e estereótipos permite que esse compromisso seja efetivamente cumprido.

4. Realizar um diagnóstico interno e, posteriormente, incluir a diversidade de raça como um critério para a ocupação de cargos de liderança.
Já parou para pensar onde estão os negros no no serviço público no Brasil? De acordo com um mapeamento realizado pelo República.org, apesar da existência das cotas raciais para concursos públicos da União desde 2014, apenas 35,61% dos ocupantes de cargos federais no Brasil são negros e 23,72% dos servidores estão em carreiras de gestão. A nível municipal, a situação não é diferente. O estudo revela que, em 2020, na Prefeitura de São Paulo, apenas 28,6% dos servidores públicos ativos eram negros. Destes, 48% ocupavam quadros de nível básico.

5. Criar programas de qualificação de preenchimento e coleta de dados sobre a população negra
A estratificação de indicadores sociais a partir da categoria de raça/cor é importante porque permite a elaboração de políticas mais assertivas de enfrentamento ao racismo estrutural. Um exemplo é o projeto A Cor da Mobilidade do Instituto de Políticas de Transporte e Desenvolvimento, que chamou a atenção para a falta de dados racializados sobre mobilidade urbana.

6. Considerar a transversalidade do tema na formulação e implementação de políticas públicas
Falar de políticas públicas transversais implica admitir que a realidade social é diversa e complexa. Isto é, ao desenharmos e implementarmos soluções, é pouco efetivo considerarmos números isolados, sem observar os cenários em sua totalidade e possíveis efeitos sinérgicos. Em termos práticos, uma política de redução de disparidades de renda e redução da pobreza, por exemplo, precisa estar associada a ações afirmativas em educação e de ampliação do acesso à saúde para que seja efetiva.

Now that you know what I need, correct the essay below based on the examples provided. Note that an essay may have zero mistakes, therefore should be awarded the maximum score of 1000.

I want the output like this, in Brazilian Portuguese:

Nota Final: 'Final score based on the student's performance on each competence.'
Justificativa da nota: 'Explain and justify why you provided this grade or score.'

Correções: 'The corrected version of the mistakes you find based on the provided information. Propose the corrected version.'

Redação sugerida (If mistakes are found): 'Corrected essay'

 \n

Essay: {redacao}
"""

generation_config = GenerationConfig(
temperature=0.1,
top_p=0.83,
top_k=27,
candidate_count=1,
max_output_tokens=2048,
)
contents = [
prompt
]

generate_t2t = st.button("Corrija a redação", key="generate_t2t")
if generate_t2t and prompt:
    # st.write(prompt)
    with st.spinner("Corrigindo..."):
        first_tab1, first_tab2 = st.tabs(["Correção", "Como funciona"])
        with first_tab1: 
            response = get_gemini_pro_text_response(
                text_model_pro,
                contents,
                generation_config=generation_config,
            )
            if response:
                st.write("Sua redação corrigida:")
                st.write(response)
        with first_tab2: 
            st.text("""
                Tema da redação:
                Quais os caminhos para combater o racismo no Brasil.
                 
                Como Funciona:
                O Gemini irá analisar a redação, econtrar os erros e reescrever com a gramática correta.
                
                A correção é baseada na cartilha oficial de redações do ENEM.
                """)