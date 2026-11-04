/**
 * CausalMicroGPT-Int8 — PyTorch-Trained Decoder-Only Transformer in Pure Frontend JS
 * ==================================================================================
 * Trained in PyTorch (AdamW, Cross-Entropy Loss) on Jefferson's machine and exported
 * via symmetric Int8 weight quantization (w_hat = q * scale, q in [-127, 127]).
 *
 * Architecture:
 * - Stage 1 (Prompt Encoder): 2-Layer Neural Network (BoW/Bigram -> Linear -> SiLU -> Linear)
 * - Stage 2 (Causal Decoder): Decoder-Only Transformer (d_model=48, n_heads=4, d_ff=96)
 *   - Token Embeddings (wte) + Learned Positional Embeddings (wpe)
 *   - Pre-RMSNorm + Causal Multi-Head Self-Attention (WQ, WK, WV, WO) with KV-Caching
 *   - Pre-RMSNorm + SwiGLU/SiLU Feed-Forward MLP (FC1 -> SiLU -> FC2)
 *   - Final RMSNorm + Language Modeling Head (h @ lm_head^T)
 */

const MODEL_BUNDLE = {"meta":{"arch":"CausalMicroGPT-Int8","d_model":48,"n_heads":4,"d_ff":96,"max_len":21,"params":67694},"intents":["greeting","how_are_you","farewell","about_jefferson","skills_stack","socials_links","meta_self","thanks","negation_cancel","help_menu","play_game","chef_recipe","explain_experience","explain_projects","explain_all","set_detail_simple","set_theme_dark","set_theme_light","set_theme_paper","set_theme_terminal","focus_rag_agents","focus_swe","inspect_chorus","inspect_arch","inspect_ml_projects","copy_email","open_resume","reset_page","hire_or_work","hobbies"],"intent_num_vars":{"greeting":2,"how_are_you":2,"farewell":4,"about_jefferson":2,"skills_stack":2,"socials_links":1,"meta_self":2,"thanks":3,"negation_cancel":1,"help_menu":1,"play_game":2,"chef_recipe":4,"explain_experience":1,"explain_projects":1,"explain_all":1,"set_detail_simple":1,"set_theme_dark":1,"set_theme_light":1,"set_theme_paper":1,"set_theme_terminal":1,"focus_rag_agents":1,"focus_swe":1,"inspect_chorus":1,"inspect_arch":1,"inspect_ml_projects":1,"copy_email":1,"open_resume":1,"reset_page":1,"hire_or_work":1,"hobbies":1},"enc_vocab":["4","a_game","a_github","a_good","a_job","a_real","a_recipe","a_row","about_experience","about_jefferson","about_projects","address","adios","agent","agents","ai","ai_research","all","an_llm","and_experience","and_machine","and_projects","and_show","angular","anything","anything_fun","archagent","architecture","architecture_recovery","are_things","are_you","awesome","backend","background","bad","black","black_theme","board","board_game","boot","bored","boson","bot","brief","bright","bright_mode","build","built","built_this","bye","bye_bye","c","can_we","can_you","cancel","capabilities","care","career","casual","catch","catch_you","challenge","chorus","clean","clean_design","clear","clear_filters","code","code_in","code_repos","codebase","codebase_recovery","cognitive","cognitive_science","collapse","command","commands","computer","computer_science","connect","connect_4","connect_four","contact","contact_him","contact_info","cook","cooking","cool","copy","copy_email","corpus","cv","cya","dark","dark_mode","dark_theme","day","deep","deep_dive","default","degree","design","detail","details","dinner","dinner_idea","dish","dive","do_not","do_something","do_you","does_jefferson","doing","doing_well","dont","dont_explain","dont_show","dope","editorial","editorial_theme","education","email","email_address","encoder","encoder_transformer","entertain","entertain_me","evening","everything","everything_good","expand","expand_all","expand_everything","experience","experience_and","experiences","explain","explain_chorus","explain_everything","explain_experience","explain_his","explain_project","explain_projects","explain_tinygpt","explain_work","farewell","favorite","favorite_food","features","feel","filters","focus","focus_on","food","for_fun","four","four_in","frameworks","free","free_time","fullstack","fun","game","games","github","github_link","github_or","going","good","good_bot","good_bye","good_evening","good_morning","good_one","goodbye","goodnight","got","got_anything","gpt","graph","graph_rag","great","green","green_theme","greetings","guide","hackathon","hacker","hacker_mode","haha","hate","hate_this","have_fun","he_build","he_code","he_cook","he_know","he_open","he_study","he_use","he_uses","he_worked","hello","hello_agent","hello_there","help","hey","hey_bot","hey_there","heya","hi","hi_model","hi_there","hire","hire_him","hiring","his_email","his_experience","his_github","his_hobbies","his_major","his_projects","his_resume","his_skills","history","hiya","hobbies","how_are","how_were","howdy","hows","hows_it","hows_life","hungry","i_contact","i_cook","i_hate","i_hire","i_see","idea","im","im_bored","im_hungry","im_out","info","instructions","int8","interests","internships","interview","is_jefferson","is_jeffersons","is_up","is_your","it_going","it_simple","java","jefferson","jefferson_email","jefferson_like","jefferson_looking","jeffersons","jeffersons_email","job","job_opportunity","jobs","keep","keep_it","know","know_python","know_pytorch","languages","later","learning","lets","lets_do","lets_play","life","light","light_mode","light_theme","like","like_cooking","link","linkedin","linkedin_profile","links","list","list_commands","llm","lmao","lol","looking","looking_for","love","love_this","machine","machine_learning","made","made_this","mail","major","matrix","matrix_theme","mcp","me_everything","me_more","me_projects","me_something","menu","ministry","ml","ml_projects","mode","model","model_details","model_is","monorepo","more","more_about","more_detail","morning","multi","multi_agent","nah","name","neo4j","nevermind","nice","nice_website","night","night_mode","no","nope","normal","not","not_explain","not_show","nvm","of_toronto","of_work","on_agents","on_swe","one","one_sentence","open","open_resume","open_to","openclaw","opportunity","options","or_linkedin","out","outside","outside_of","page","paper","paper_mode","paper_theme","pdf","peace","play","play_a","play_connect","play_game","portfolio","portfolio_projects","profile","programming","programming_languages","project","projects","projects_and","python","pytorch","pytorch_and","pytorch_project","quantized","rag","reach","reach_out","react","real","real_model","recipe","recommend","recommend_a","recovery","recruiting","repos","research","reset","reset_page","restore","resume","roles","row","school","science","see","see_his","see_ya","see_you","sentence","sepia","short","shorten","show","show_all","show_help","show_me","show_projects","show_resume","show_technical","sick","simple","site","sitter","skills","social","social_links","socials","something","something_fun","speech","speech_corpus","spring","spring_boot","sql","stack","start","start_game","stop","study","summarize","summary","sup","surprise","surprise_me","swe","switch","switch_to","take","take_care","talk","talk_to","tech","tech_stack","technical","technical_details","technical_skills","technologies","technologies_does","tell","tell_me","terminal","terminal_mode","terminal_theme","thank","thank_you","thanks","thats","thats_cool","theme","there","things","this_site","this_website","thx","time","tiny","tiny_gpt","tinygpt","tldr","to_dark","to_light","to_use","to_work","to_you","today","tools","tools_he","toronto","trained","transformer","tree","tree_sitter","ty","typescript","undo","university","university_of","uoft","up","up_hi","use","uses","very","very_cool","view","view_resume","voice","voice_project","want","want_to","warm","warm_theme","we","we_do","website","well","were","were_you","what_are","what_can","what_is","what_model","what_technologies","whats_up","white","white_theme","who_are","who_built","who_made","work","work_experience","work_history","worked","ya","yo","you","you_a","you_an","you_do","you_doing","you_feel","you_good","you_later","you_today","you_trained","you_work","your","your_day","your_name","youtube","youtube_speech"],"dec_vocab":["<eos>","<seq:greeting:0>","<seq:greeting:1>","<seq:how_are_you:0>","<seq:how_are_you:1>","<seq:farewell:0>","<seq:farewell:1>","<seq:farewell:2>","<seq:farewell:3>","<seq:about_jefferson:0>","<seq:about_jefferson:1>","<seq:skills_stack:0>","<seq:skills_stack:1>","<seq:socials_links:0>","<seq:meta_self:0>","<seq:meta_self:1>","<seq:thanks:0>","<seq:thanks:1>","<seq:thanks:2>","<seq:negation_cancel:0>","<seq:help_menu:0>","<seq:play_game:0>","<seq:play_game:1>","<seq:chef_recipe:0>","<seq:chef_recipe:1>","<seq:chef_recipe:2>","<seq:chef_recipe:3>","<seq:explain_experience:0>","<seq:explain_projects:0>","<seq:explain_all:0>","<seq:set_detail_simple:0>","<seq:set_theme_dark:0>","<seq:set_theme_light:0>","<seq:set_theme_paper:0>","<seq:set_theme_terminal:0>","<seq:focus_rag_agents:0>","<seq:focus_swe:0>","<seq:inspect_chorus:0>","<seq:inspect_arch:0>","<seq:inspect_ml_projects:0>","<seq:copy_email:0>","<seq:open_resume:0>","<seq:reset_page:0>","<seq:hire_or_work:0>","<seq:hobbies:0>","Hi!","Try","'explain","experience',","projects',","'play","connect","four',","or","type","'help'.","Hey!","You","can","ask","me","to","'do","something","fun',","Doing","well.","Type","'help'","see","what","I","do.","All","good","here.","Want","play","Connect","Four","check","out","the","projects?","Bye,","thanks","for","stopping","by.","See","ya","later.","Take","care.","Peace,","have","a","one.","Jefferson","double","majored","in","Computer","Science","and","Cognitive","at","U","of","T,","researching","how","AI","manages","context.","He","studied","CS","University","Toronto","builds","multi-agent","systems","RAG","engines.","works","with","Python,","PyTorch,","LangChain,","Neo4j,","pgvector","AI/ML,","plus","Java,","Spring","Boot,","TypeScript,","Angular,","React.","His","stack","covers","alongside","PostgreSQL.","GitHub","LinkedIn","links","are","right","above,","'copy","email'","reach","out.","Hey","there,","that's","me\u2014I'm","67K-parameter","Int8","Transformer","running","here","your","browser.","me!","A","PyTorch-trained","causal","quantized","8-bit","integers,","locally","JS.","Thanks,","appreciate","it.","Glad","you","like","Anytime.","Got","it,","leaving","things","as","they","are.","Here's","everything","do","on","this","page:","Let's","Four\u2014you're","Blue,","I'm","Coral.","Click","any","column","drop","piece:","Four!","Drop","blue","disc","into","4","row:","Braised","short","ribs","over","garlic","mashed","potatoes,","slow-braised","red","wine","beef","stock.","Crispy","chili","oil","scallion","noodles","Sichuan","peppercorns,","gochugaru,","black","vinegar.","Pan-seared","duck","breast","crispy","scored","skin","cherry","port","reduction.","Miso-butter","mushroom","risotto","roasted","maitake","cremini.","Expanded","Experience","descriptions","below.","Projects","all","Project","descriptions.","Restored","one-sentence","Switched","dark","mode.","light","Applied","warm","paper","theme.","terminal","Highlighted","Jefferson's","work.","fullstack","software","engineering","roles.","Focused","expanded","Chorus.","Multi-Agent","Repository","Architecture","Recovery.","Speech","Corpus","projects.","Copied","jeffersonn.liu@gmail.com","clipboard.","Opened","resume.pdf","new","tab.","Reset","page","default.","Yes\u2014type","his","resume","above","get","touch.","In","free","time,","he","likes","cook."],"encoder":{"fc1_w":{"shape":[32,546],"scale":0.0082536,"b64":"x8rAxrzMRMdXn1xIsvkgIyi+xa7KrKxgv8FfTVG6waBhr2JbWcXBSrdcsbFcVlNjtLO2sMNYVV/CX7HCxrBctrpPVLi8t1VSzcSoWVXJycfAv2tPVFRbqU9TrGeyY1NVxb6+W6u6t7RKR1e9SsMSRsbBUlRWnx4drmRHo6O5u7a2ssTBu1mrXU1QtFFLS1O1UbBMTmK1UUZIU8O/x7ROSmG+va6yur/Buae8ur3ArK3CyMcfJadVUqhfX1VUpExLwFS/TseywsXAULK6vVertLuntLq4s7alXE63UsNWRLtJqFzAxqG8xsFUVFFNskJF3LhPs1BZsV5eo6hDur27vV83YkfAQ0u2uGPBwsbAz7K9yMnHv8BkVFpCRLiturZQUMakori4vLfBxMbGc8VSUSKw+a+8XWKfmFnBx8ZmIlm0uSwrUcAiX6W1WlhQUGJKTE9hzEwITbq4KVGwWrdexBBPR1EqIB1srbrBxsNMTLe9wlhXqbeTyp+3JlBYr83PWExCTqO5I1hUYl9cwK7J3EXDxr0uqKg1uFixUlbFqLC2Ura4u6a+xa6rSkaxucPAUsOpsKS9vFpVWsDEzca9xb29vr/AMzNcU1itqqS6ulK/vbQ7qVHHxrqsVFhSr8fEyMPJy5xNU6O3ZM7Or7e+J8e9t0dITlHMzCMdxcazxsXMBwC9x7+6W1rCsctRSVBTwabNyMdkyLm3x8DOpb+/xa+rxsrHS0fGuse5t71BYFFLQTy2urHKsbxbwL5hVFZIN6Ndr1yorcXCT7FgBq+pq6+ovFNQrsywS6dGoq1CQ7BiurhQULzIulNOvsWspKW9wMDDxUlHQqShrURDTmpaprGoUr+3YbK0uLG7tKK9TMuivUdKTFNMo0lHsk1DV1q+wU3/Yru7t7G7pdNdssXSsLK3tGO4vaVXV09Ip8HBwq5RTGG/uLKxvsVHTaJPRUhLXWHJy74+Pp2usVygXLa2pUREvrjJssxNuMbMr05MSKlOSktXUVFDTkZZSLywWrq1Tce4XFhWxFhFR0quRq9KS02yTbmsW0OotFynVsM/TkRLxlrdTbxJQkRGRqbFxs7I0rpTy8fNwEeZp6a9wrqzu7e1tMCdoUlGuLHEw8HBVbi2uE2zssLDp2JgY7Yut8Nhtb+9TD86TrZNW6Sys7BSTWZNTk1dyFY5SDW5U1ZJZEujwFJQVlVTSkdtWb3GxL2+vr7LxC2+s7Vg0FixTElJtsHDpbq4T1i4OWBUYV2jwq+7TUxIQrtQq7IBvKjACWLDnrO4UbvAvafGw1VVT0y4tr/EUsCvrFzAt1ursE1KQ0bFwrW8ycjItLupsbWiop+zs7NHRbkmoVO9wrewramrRUZIy8bEwr9QUZyzaMbBsFFNtMm2s1FVXFzGzkZExs26RcPEr7Aaw81NpqCuw8e9ure0Slw0w7mkRVNMQUnIq0JKrFVWv8hJUbHISMiosV1HXh07MS+6vabDr7dbwcdlU07Bsl5epazb2MO9TrVQZ6tbYE5ktFlTqcmys6JPralMSrRPT1NJQVG4T1VVusKppqi9vsPDxFZLS1VZWElJVZdh3enaxr24Wq5KsaxGR1m4uMWoScbGtre1ZFZNrVRAZGO9v7Omqru2vKyysRBKsrm+U1W0tWFJTaPDRkZKVbvDwq64tFi+vLhSTka9XmZQsbpVXWLGwr4yLmOpsa6pVq6pYLu5uVW7ScOxvcLEtrO8uKaxubursri6rq+KTbhSuLxXu7a7qqrAv6rHysBHS1K3rbJJVbxIVkumtKWrirJDwba/wlXZUUG2QkatqLDBv77BwLJYusHFyMljYF0/RU1ZS0+8vbplXLm0S1G+urW4XLmxrz+v27TColdiX1S6ubpiBGW1sjI0srw+q1xU6OS4slW5v7euxb0rRQ69tK6wUqumQUu4vkhWUlCYYsLHw8NOUEi4v15bqa94xGGzOk5Nsb/GVEhFT4tPLE5EVKqwxK68R7RKRbVdrrDMu7HCVLC7Y7NSU7JPTV/FyF1cTUmxs8TGt7atr6u7wVlPUkxJRkS6u67Bu8TA8vOnqqtbXWJQUla7wUyxY7zCvLGrz1mxr0bIvr3DwgBTUWGrVb3DqbW2tMFTTLi2QkfGx05NzcRTxsLDtLa3vsKxYFuvuLO3vMK3Sag9xsOsy764SsrAqrrCuFVaR0JEuLNCRUCxUZy9p1BeS1JNSVvIXVhYR0W1y8c1U6RXWqheXkhHRVWktFpgYLT5Sq6wpjyquaa3qVq/vFSkubxRVje2S8vPS01Zo6lESUJHQbG+t1JXsL+4tpqlY1lZLVBQZVa6U11ITWFSvEU2RS8us7u2pLewXLe+qqhKRrlbT1JSUvdYrEOsWLM5r7hNsahKSaNBTkhLVERIQqK0sVxISFJWSUQ1x6+7uLuypaxARUVLS6qttLCjqKyxobq/TLOwTcK3TL3BtbS+w6Kyv8OwtLHDsriouORMqEavuam/s6QnQq80MjNTvlC3trRMSktLqr+kV6Oqo1e/O08xRlhRs0S6vsSxs6xKSbu+wqesukY6PzJmXGVBQUZTSU2ztUOpqrG1rbO3uU9ErUy2r2NYXltBnV2qqUJBSEO6WJxVvFNTtEtep6e4V1m5tWS3ur2mRLlSSk9QtbK2obShRbG5ulGjq7GhpElAPUi3uUa5u6raYaWlwa9RXrW1pUJAXURJy6FJUGNXaaSpQVVGtbm9vkmdWFs4VK1SrqZMp1Ety6ZMR1ZDQ7e0TkqppUhGs09VXq9LTVVaW7a9wL+7vllJnrm9W12tq6+rqKWyu/S/Ly9UpbpGRlVaVlqwucQuv79GQUfQzaSpYkhHXBOasbu4ua6zsLY7Qra2RUEuLUFBSvhLTL8JZF1PTUfavsG3ta1BREegJTxCvC5CW08qTLOwNzdBUbc8QjS0ray9XENZUU9KQlo+XVZcOTafsrC8IqZVo2CsrTo4STqnq2GporSpvVFQVTmwT55KqF9CSDmntLq7tkpLRrGyurlenqO7wDU5Na21vlhWsL68t5NWoK6lrUlGq6i0U1FHTF9LSzPoRru4UFNTpE5Mp62+qas+PLNRoEtHSNxZpf2uW8lLtrRRsVlHR52zv0ZKVTg3OllNTlo6OTxQSEG1raxQtLZQXFw2MUZLUamws5+nrK6qoktMNLVLTEK1sEZHua2uu6OltLadpqm8t7KfsPtNWbi0tEu1olimRZ+2vrNPt1BIsbFKUEFKT7qhVVulprDGtDe5R1eutj6+v8O5tKpKTkhHPlFUSzkwO7egqqRBSEdWREewskihoMC/uLRGRsG9prOssl9cVls7oVatsK01QkqcU6pLsEtQUlJZZai4sKlSVrBOVUhdxFFPSExUs62zp7KnRFJVU7hcVkydXDcyNTS8vUhOSK/TW1WkQa5XWrSwWUFGVkVFs6RPT6u7p56rN6bARK9KRk9jW1s1U6tUuqhJqFi2rlNNR105OLW0TklbTzY5VrheWKI3NlSpsUdLRj9MSVVOSEZMEA+rrK+rqqG5s8i8t7jqpU1DQVRbrquotD+0RkK7RDuusKhUqby/pqy43UuztrCturE0MFFUOTW5v0FEVCrES0mspKJWv7z3trqxS6YhQ0qturKuSbs+YzSzTrazQ0JBULo8SUW6YMa9WrSkta6wQqpHp7RiRUCjsbNLUalcXqisq0VASktjplxZWb67UFFOXkJMtl5GrVtEQ01fsbizuFFVTLK0TERdV1ZGTUNCPqu6t0xTqbe+tpRdra2wTa6znl2zp6BER1apt0FbQUxIt7O7p1NWVrG9rK5JSKy1ZqyqsrCkq7pbp7O5x8ZUtF1JSFtNsA8NS0JHQlyytl5FRk5RSUdTXadPsKxQW1o7Qkq4sqZUUKNeX1hWpsHAScNPSUa7UU9HtKi7t1exubiinrS2urmktLpQsU++tFW1qKZKQaVJRU9NuEi9saxFUEZHTbVbV6OspSzCU05MTWFYsD6zwMC1t6VHTkJKR19TRkI8QVBeVlhFQk5SREpQTkahoru0tbNMS0dJrUxWUaKmpbBDWl6sqldedEyfq7GutLWxuE6fqKS2s7azuKC7wrmvR7OsVU9OrKu6YrNbRE+ytLJeWlGaXkU+Q0fIwkxIT6qtoFqlR61co7CxYD9DTUM/tKNGs6a0oaWnP1lJS7BPSk1jXGCrrFXirqKyplnUq15ESVs+R7K5S09kWUJGslBbYqJFS1tUU09FQ0JNSKCwa0ZOs7lcVVKnpqG5s1e5T9UdqbVDSVhjrFtWvUZHTU5DQCu0s6hdoUNIXFiMTUS7s7GuXV5CPVZUQjrEQj9AOlRRQ0hpXF1LU0y8vLuxSKRPP0lQRUlXRkhFU01JSK+wMjhDtlHAwDuoWFzDqlNmT1OyvKY7pqugNTOisLG3rV6iXFtRVzg2tTpgV1+trElYRquxWTqsUqO3ol/AwTNaTku0tkxKTrGwRUZnqK1JRTo1Na61ua2yWLO6sG+sXFVStbC1o1hRs6S/uq24RzStwbi6T1NQX1BPXKzCsKo1OlCrpbSxs6Shqe9Sp7SzTE6vs6y+vKuytSQrrDkyOV5US6cxMTlSRkCz4mG2U1CvrKU1MrZRT16wsF2nYbKzWkNHNlFSuD9YUUhBslFLTaZYSE5gYlNKVlVytqtQUVBQT1exXVe5vV67t7u9tLdJXFe3sz23sraqpF2gc1jAsLWwS6dZscFRvcVSWqRRT0FAP1WuRzk0M7Oiq6XAvkdXTU68s7paYVZYSUpLQkNIn06ytGmiprExpqCfo0rMrbikpG+uWU9PUbNsVVhLV1dOTqBJTkpXSFJfqFdNW1taYlqmQa5UVK9ZT1JurDc5OTpLRUlHSldTp1ubPqelZ7OzWcbBs7y7snRNUaW2pl+gOVhJPVSzvFBdX2NfrKumVF6zX15FrmBPRlk1Mqyzu7NcVDo1S0tiZFs5PKRNTbi8w75MSq26b0lC/v+nq69WWmBLT0lIs0tDXlC3u6tlV6WsV728QUBGxKGutGFgo0hAYL51skNPSFpbUlUyNFBRMjlJu8K9OP7Ku0XdpK2tSUvHs7StulquvsGxv7qtu7fDq7i0tqyuQUC8trnFxENjtK1GqkpoUVCxuaZFo6inPUFgSU9M52ChrFSlqUZEulFnVF+ur62itKmvYDyuTKq6W2S4vlVlSkhTVdFOtU5Nvrlao6a6vT5CPVxKTLCxV1JPp26qorGjSbWxY69Otqa6vKuxR0SjxklITUtPW0xUrlpJo6NNSEXBXbKyuFqkXkFaqUw5r663Vqq5vqdYUPDttElERF+3tKdIRFKztr9HU12zRkmwqKc9PcFPS11RVlOnZFRYYEJDTLJRs0iwt0xCUkZKRalHRkFQTklCtLSiWk21r7OyVVdMVKVYxFNCRkW9ULRFtVTBt0y9rFGlrqRepbJGU+RMQ6PfXr69RUqwsltFR0BKP1isRz1CRkymsaTGw7astbu1tMNbXrq+T1FJRr24aLZVUm+quqVDo6amnt6+s7xfu2G1SE9OTrZrVVtMqatMS19ESEpavrJdtSBSUVusYrKhvk2ysFNWWE9xrEI8Qkiyt7hOTqOopFuSQaCnZFJQWcPErcO8TJ20TmFWZGVcRK22BFS3uUtgYF/MtKoT4V69W1lUSV+5uKxDPaylvLZgV0RGS7NhYk9NSKWts7a7u8BGS7S4ZkNG/AReWlpaVFxJTkJDTFE5W7a9u69br7SqrsNIRke8x55LT1laYr3Bpk1BsEhHTVRYYVtCPU5RREBHQ8jBqS00uUlQrauvu7wrT0pVtFHLxr+mSVhPukXBpedNsaqnSkRBuETFQkxWUzw+qBNUT060v6XBqqpTUEdgUlS+sqZTXKirpk9RTWpgr5uopz5LTamxpkCrsKW7XJy7u29Vu7axrkmuSFNURUyboadITVBJSktJRFJWqENMT56loquqtbq2oWG4tq1JR124sj+oRb27tbKupVRMWVFCV1NbXr2ppbm9tVKsXE1Uq086PDmvUKpHRZ20sExEVVRKSaROUFZfYXZRTki4uKm1wMC/pqZIRcBJSqWzq62lXbSvobi5WT6wTMJPULi/Tbm6w6K6ur+ttLm/TlFdUUpNVVA3vbNIrFeyxbK9vLhMR1S5TrVJTVBQpUaksF1aWk8/s7K4tVVVUERFQENQS1qvt8K/v62tvU09TruXqqZIREdPS0i3uL+gp01HsLG7tkxLVU+utF6ow6tWo1RZVaq3trxf9Fquv01Pt7Rcp66yra24taO3tbmrSkxFTLOqBqdKXlChSEBRSbBaWE+bqlxDSU48M0i7t1McrKdbv1GnUUtKq8TIWEhLTFxQS6SznqVbS1xHt7q7u6tcn5+vuKvLM6/Fp5/uTq1JTVVMSkxRSUilpkpRslGbnrJbXFeorrW+wMK7uLS6tL+3MDGsr7CpqqS0u8q/uPZKqVO6trKZsqisT8HBt7pEx9BTUqmln0RHW7S+r8C4u7GuVVRGQ1FNQEbSv8HGUa9At7e0pKSxSUtLTFBPu7Kzw7yovbm4wMHBqLS6uE5Nvr9BvTtBu727ZrtAslVcSU6sRK4sqMViu7xhVVS/VKlhXa9RVcC9Uazht8K3traiVrq9VsG6uqS9sMTDwa3vvblVT1NAR0xWT0XCo6dMTcLAxEtKRqimsEpGUZeyXlBOt66zZV3Cs7C5uKO1vsVEwby/ury4rlJLWlI9VUy0tciqrrK3tLe0qsH+rLi8srVUs661uaK2UUtMor3AwlQrLF22uKxVSUW1srG/wL/Esa3IwE1QS69RUbOl4FZUrcbEt7ZFsjVBUjw6uLbCwaq0vL+turPGQDlPTblPLFO6wkG9szieSLC7vsC2S7HGOcK4tr22uEumXDirS1VGqk+9AWFdTLk6REY8QKcEBTk0MFS3PMLIv7uyu7C+wE1PTka8vUispjU1t7s+OlJIWVNYUVipscPHolpfXkxeaUVkr7qsxFNUt1Bhq667Tk++u2PDyceySTBOScPw97Q63zijRRMoLE5WU0yWs7HCwMG9uU0/PK63rltdNFZeW01IWEZGn7u8UEtKT19RZqStwl1KwcK+w+1axMaurLO/u661rMbhTVxITVe9w1FNSEtUVr7Bv03Hw6+ytFowL8G+xMY6O62veUA3xcZcWVS0ta+5u0/EvOQ0qShKSlzDVbq1Qci1OzdDQmNPUaxaYUNEXLLIuzm6wLe8+PjBwk5SwsbPuUE/WthLTD60s7NTWVI8tbS3wKsRQ0Wiv7a0xLxGYleuUFJKwsQ8wEDcQ8UdsbdFrlExKi/A3r3JucVcwMW00NC/tmVVrlQ1OcbFRbVlWatYVrSqwrC4rsdZTF+70LDGxrBaTlBDQPq6Rc/Pv8GvYVjCw8LEyE1KSVhZYUxEUpKoQTY0uLzAU65RwrRGTFjER8UaRsXGSlJMZFJNqFM+WVq5vFS4ocS7wbu6z9leujjls7jX4q5NRl+3O0hLV77GwbFUTle+wrFRP0C7GF64WVe7p6/Fy9QvK2Hu6WFhZPTuZD1Evrq7VMZCuL2+4FtKSV9dTk5iX1NJPz1MT/JHWLu5sLXgZlq74mO8vLxLRVNBQbdJ2MFQt0NgxF/MSq9Ht729wVZDVElBRkA+O9DEvMe/yLK0wcPOwr5dVV5DQkVRRUNLUOFoYzw8UVC/v76+Wrbx7DO+tcHHYF1ZYFhdJNK4vsG4UDMvSN84UWJNNTlDSk5DTEJVwFMmULi1w6xAYz9jOghPVTpeU1OZsMLFxcTAvEO+vf+6vbJfxlfENEtKreDjVEZGzVFEM08/U5nQwai/wbG7wbpfp6rSvlbIxarGY7Mo1rhDSFjKylRTTkessr/FSLepp2LBwFpZWL67y8S7wLm/v7zCvLvk7fBfXGhOT1ZPuSXPYU/V2MiuO1RWP8LDwcLE4hjPzWCsTMG/rdd7HL9ST7OqYlrFyU5NzM0+w9/dQe2w1cEZWV7TwcBF6+TgwF/h3N1fv66twLvbtLzB3kxMQzu/urlDS0Cys2c9pC9eS1GyS6jFrLVYPzy52Na6HaJbpl7Iyz5ASktcsGJaWVU5t7G5qUBXV2K3pV7DxFNWtrtWUrG1vdPSwbtmXVm8uUA9PE5BP1BVrEM8X5uousrHvbq8Yau7s8ZOTVy5UTdPScLDUVRVoK6tq0k3ZGJDSrmtsbe4tbG0pSFVqMjKV01YtKhITlqzU1BRV0Y+OalRUFlESlSrvsG9uK62vr+6pKc3O0hSVKBTU6ZhWFRSpElJS1SyU8KztLy8t661vFmvuL2ltrS5ra+USb62Ubhatq68rFStRarBv8JKP1JLr7ZNSkdQpj9hYFusj7M5uwq4SVu6TES6Ozm1tKpKUcXCxq2uvDk5QcFgW11DSLqsuLdJTk+lpbq6uby/vMLAT71TWl+4ybs+XFZgaVpnc023xCe5ulJQVVRgZa+8yshXWGRRUk1lw0xMRlJTq6yvXrBcwv1NTk+hqK6ZrEpAPT1KS7a8vFhdqbF5xGZfYEVJq0dGVkZL1I65TVpSZaSnQKjAur+5vVKeX2UNtFPHU6TCoF291LC9u6dBPFlhSUyosUY9U7RkZqhHRFQ8M7q7vsS+t6+7vLa/ztVZVlasq6i4tVbDxLrzpk1PTF9ev1tXrsLDvb/FSW7T0qinZsbAq7q+WMC3tLeuVlA5OrWsOD+7wUhDTkLbUby2XVxcvsH4urq3uqwkSEhkxLG4vbxFYwu2UltgRTtCsbm6wT23t+pBqk1lUkyxuK4/rrSlRUadr65LFl6iqlZRVz8/tlKqWF2nqtupv6mwWztYSGG2qmG9uVKvT0y5uU5JTq6zu8BbX2C/ukNGQmFSU7CxVE5PpZajV1NSSrOzqKpKsqPAvq2vTUBSyEdHUVVSWFBNqVxDqqlLR0jJXrC2t6uop6+xq7m24eG1sqW/vV5RuFZWr0hGPWC2tZ5DTFBVSklLWlivTEmwp6E+P75OSVuzrVxjr6uyXURJStVOukays0RJtE1ISFtVSERZU1BMtbGiWbRNrbPes1O7W6tbwlxLRki7TLZHtKy+s0e3sFFeqqijorFHVC9NTqLFXMS7RUWxtaRITUJFP1SxRT48Q06tq6TDw1BWRk1OUbteXra1UlJFTLm3a7Kzq2K0rbdHZaKjpkMHrbShs7SyTlJNTKpnUVdLUFBISKpJSkdTvrlVo0RTp6u5qq9kSFCxtbJOUU+WokpDQ0Pe2k9JRKyrrVeWRailXVRRXMPDrcS/sqVMTaixqpuiQqm8tbayt1FUW2Kyr1wbGaq7X1ZOsmFJT1pARqals7BdVkJGS7JaYltKQ6FUVLu8vLpMTK62bENGr7OnsLNXV1lKRj5LSlXAX7i2ua5eUapVsrpHSUTBu5uytlhep7u7rFBUUkpITa6otbc9QElMPDxRRLvAolEouUVPqaqqv7mutbG0u1pUw7pfSlFRvEu/nDBJrKmuvLxITLlBRsO4tbzEVlFXT1JKQVlDYlhXv7pmT08wTaZZq1OttMDBRa6iAqWrp7KpvVJMYL6oUqZDqKlGPaarubazs1ZQT09Tt7+ioqO8uby8wbG5v1RUqbm5s29Qp7awL01Jpaa1TlhES1lKS8QRRTEoT1JNpLCypa6/qK6yuktVOUdPSb9cqsuzW7r5tLZQtFZMRKVDt09LVLXAxF9UUlm1sKhaUEQ7QpxERklJVlfAxklLS6VZUVWeqlhVpkpLtrdWT0KzuUhMs1NESaNRS0dXUkhEtrGkuMlRVrC/UUy2Vl4qRlMuLjFMvklEuFBNT7BNR7yfWGWoobbAQEs2tlWws0C8vr63sau+v0hIQVZNTMPEuzajp6pHRUpVSku0sEKiqL+4srJOSsO7r7hVUmJaCVu6pFivr7hHUUNn8L1SS1JSS0tjW6a0srFRSp1LTkZXv1VSSxKzWlexp7SiSktVVbClrrRwU7jCwsDAuk5KS6q7XleiQ69bX7i5W0E/UkNITqJPUaWyoWWrvai3RlBFP7afpaRQVakLtGBJoqu7U1FQTl/BvrGzRU5fWLm5T7KnqVG6slO1tUpKOkBHTldNUU5K09RcUViko6mys7RHMb3HpEtDSVijtq6wtT4vQ0vERThQUKRanr7EqE9So0e1sVBTtbnHwq2zwsa1LD0+UxkbTE9Wp6ZWvLrevL2yQ1FGP0mhMkUtRCpCXE4yTbi1ub5AS0tFSr9aV5xEVjOuurhQSl1AWFaivL3b5uq3QaaiWFmzrLW4u6is+XHi5K6fU0xQWceuUaNLWm8/R5uosrO1t1VSSu3nRElsqalKSLi9uFFIRV1jpUxFTJhYp7axsU1LpVS0U1JPR2NJRsPwSLu7T05VqLOyVU1CUlm1rlA8m01GSFVYXFGvUUtOtbRSUFlIT6SysrW4XbK+vVivtqWtrJxST0mxGZ5RTk1FVVa9vkezuaO2tVGgpbm1p0lIs65LVUBUS0pDUElGRapOR0VYWExBTlNnTFVQr0qlqktNUq2nQ1a1u7tSRVBNTLNLTrRQS0ugUqRcZ01HsUq5T6NRUkhLR0RQTl5QTj5BPFNSR7nEwLni5eNCQ0pbR0yztEWhpExHtLZCRkhOWUe4u6lXZVq9o6pfYbNERELdXERPSbaxTk+lWaawrrZPT6pLRE1ZSbO8tVpTpahNqkugRlCxr7Sss7ObWKy/vre8uU1GRra8V1hgQF5WrE1OWEE8W0xI5mZLta6xpKVcvFtINqtGQ1Skb2mjUa7+u6tLoWz37E1KTmG4v1ZOsrJVUbe1U1Fwb1awrqezskhEPj5GRldHQENIXWKps7ympqaws7BGuPBLo7lER1Rpt/GrVES9SERGPVzs5aRWqEVIV/BtrD64sbCqtrO/xrSzwMUnvUJDWtBHREUZ6+JST09YSUpQSFMiPkWquq6wP7dFYUS3TlZQuby/sbRBvb9HSk46ojXO3uFLSFRGVU+uwbulsrNJTqCmXF+xqMC+uqqo1qqrsFJYSq6uWMCzUaq2WKm9tqeqsrBUU1ZQtrK1SkmlqKtJTLq5wE9DSa+vp0VGX3ClpauuT1FJZ1ezUF3BuK5QTMEOvkJFT1FPplxkV009Zmi5sUdcXE9OUFhWWkyoVktPUk5TT6q+u6tVTsTIrri+u1vLzq6xsKu0vcFRQKGqQECtp6fDxETf5J7Rz0irpc7LpE1MtExYuUqtTU1KTkhERq1GSUdPQ0hAs6uSS0q0wUlMVk1MTLtSQ0hDSE65RLNMrU+/uri7tUWqWbhdk0Q7S0pQtqk8ScS0QT6rqlu4tkdGQ1StR8G+ukumrrDCwrqstru6vESnpq6tsq1GSkhKUk7Ly8tWWVK0qaxra69LSUigUk1QR9XTUUbLYqivrrFMTmJGRUZhR8RSvLS2WFaop6qlwNLJy050YldwqrPDvrtHSrpJUGpZXVl3RGhVyk1JVz4+s8PArZa23WFPZmhfvl5H1la7vLh/qKlRUas8Tl5Koaf9s1C4uam3vVtctbVcV7+/U0+pqEi5tqyutLq2u7pOSFRIUU5LU03Cy8ujo6GxtbpCS/xEosVLRFmrsa+0qL1LTEhEQWGyraNbaEJIWE4mxkivtFdUrKrCyGJdwMIhSD0/UjVKRU1QqbBORElNSklNtE9HPUemR1NXuUM8YE1LRFpaREFCtlBCwTyyqq8+s7WtvMVLSl8+XFZnRUNiU1a4T2BmpmRPVT9DUkqgGlupprWpt7W7WTqzU6bCrV3GwlOnUU23t0lRTFROt7xYqbK0uUJFQDM5PaafVj04UpuqV1JXs0dHoqdPTlC0uaVOTkEOwL68VldWX1NLpzY3V1lHQr5UqkdHTOxerg6vVdBDs7dWuqu5vKm0skJHpUREP1a2uWNFQ1JUSEC2vmO/ur66q6o7Q0vFv2GvtaWnn7GsXE1RS7hLskBTtE1Ctq63u6y1uryrt7O8S1JfOQNRqq22s025rayuRKvCv763OLVLS7izuUezskCqWqapYaE2siy1R2esPrxROjhPTq5IS0hJQ1KzRz05PbqfqqfCuk9WT026uEtcY1FNTUtHTLe2ObOzsapYPVc/rWZgXupJTExnNqxQvry7VlKsYVdPVFVOTqNTTEpiwbuxVDJTGqtKp0ypSL21u7hYTkyVq0I/O0a1tUdIS9cVW1RjRGBZsEhEWz9BpLq8TV1MwKyxpaSpQaG5urO7xk9aXF4gTK1KsqVCWlNOUlJOUF8/QVFXUFFUVz9GTrdYV6xLRmLOxcC/wshOTVBNT0pIEAmntLNbYGFPUS/Bv1IHYrNKSFJfWLKyUci5SkS8QmFUV11Xo8HCpLK2skJLSbGurrU5PVRROT5PwEdGVzzmS0uzpKNSu7UMub+1vKcKQEOnv7WyxbtAYy64TlBQQEJFUrdCvT+lT1VHYlphTkxYSGDFZl6mQ0Weta48UqCjVqpVUEpJuVSgslupqU5cRlVXqUSquKdMoWNGRFuntrdST7m2Sq6vQUZjqaxHQ0lGQWJMT7Grp1JNrpxjVE5TRFNVYlKwUmm3vapUtkGyu0E7s7Gyo1FLVmNJpqVOSbdaWllSUapmpUOmXK7kU1BOrl+4tqJSUWNkrElERaRNTp5GSVdOSj9KKalRsq9MXGBGQ0pQTaFST6yloVZVpbfASlG2s7myTrW5tqy8vamyu8CmrLDBuLKlWbVMU0tTtrC1qlxOQ65AOT63VbW3uLS1VVCvV1KkVWarnk9KTUtJSqRVYb6yQ0KxtahFRsG8wK1ZukU7R0Kpqq/AvEpSRku8t0OgoLe4rbO6ukNAbk1TU2JYU1xApp+doE0SQEiiVitSslRUtk1iqKawVUu1umC9uLmnRFNbqU9JsbKzobWpRFhSUExQSk+eZU5CREhGSEO8uG9Waaucw6VSYlRRoj9Erru4taFGUF1UY5+lQFZHTK5KRUpYW2NHWaxTV6BSolkLrqRLSldDQ66tuLqipkVEsU5jY6dLR6MsKE5HR0m1u3JOnbq1SU9cVFGqqqews1DAPwpEo1NKQ0xhU6yztkM9t70+Rsy2rqGiZUU+TyKYsr2zuLCuqqQ9QUpIRELXPTw/DrNJRLgPrK5SRUnBurOzTKU/RUerQFZTSD0/W09FSKetRkBGR7pDRUhiXq/GVkGus7qrSKvCpa2oQUWitbW36KelS1unqUpLtFOp62GurLOhS01MpkJVSmJBWmM/Plutura1tSeoTbS4OjtbW145PkNDRLO+uVldsL25t49Wq6quuayyplG1q6hNR1+uRUc6Qr3AVE1RoK6wUq+/raxQUFKjp6uyq1qjXO+rqFA/tq9TUlJNR1ytrbO6WE5CRKVISKVLUVxVTUm2y6BPT1NDUVFFQkW7t6lTU19brFBUoUVMTLGwVra0Rbq2T1BMSVxTSUZZXFFKsbWku1NNUECtqa1KWlitRmC2vrlSuFFIt7BOT0lMSbVkXFtbqCy8rtG1RaJcs0e2v763tFxISru8xaVNs0BFRrakrKlHRUdRUE5JUUOoqLm4srO4uUhErkBRVaerqqdJX6GvrRBacEaiskCyTrO2TE6oWKy2sKxMSp9PR0paOFC5tVNSsK2xqLNfRypRTLWtrLCQVUo/Q0O1tE25t6etq6Srwatdp7a3nUFDYEVHtqNKtKSwo51bR1Q9KqhBQU6oXF2usVwYsKS2p1i+saZGTVpDQbiyurOnpkRMS0FhXltSSaavrERCPUK0t6e2sbm0BQJTV02lp6CysylDtbxNpU1KTlhdsrZWtD+1tbY2P0u3t6Shoz43VMBwWbizuq2vs7BEQLWxRD7SwUFHWUPcSLTbpaReSkFXTElPSFgPQkZburKrPLpGYtW8V7GwRUS7vUxGRESlVGJCrrWrtrZOQ1pBV1G/SUZnUFK4C17GYqWvrUlK3GBiOZxcYFFgQ662XUWzuKO9p5y7u2tbUFS2tOxRtkxVTUeYoqxLSE1JS1ZHRGBfXEhNPpamqaquu0pKoF5LVFZLSWhOu0bPRcK/tbGsYLq/XlJHQURVWb1JqU5NTaxdqB1UWrXLVFNKtqxKTaa4tMbHYVVMRWO0tsFZWWqqt7y9x2O2t723rKtBREi2tV+zra6pYrWtYL65U05VUUlMU01Nt7S/vaSuu7qmrrC+S05oTbK0p0tXtk6/qKqyRK24ur1TTFa2UbBLU1NPrk2gWqiqX09ErwC2tclMT0tLQENST6avs0dNRVu0SkQ9RrpjX2JKRrivtrWyskhgY0xPT1BOS0xHV1O2sKpTMlBRoMJPSek9QElmL9lWvbO0s1OrqWBWrq+3taC+wrunRLS03LKsNqlMW1Cjv0W4uq6zt76Wr1NHQU1LSrZQTFlXV1lZSERWrlJPYEFDXURJTWC1u6ewoKesTF5FuLW5vbComZsvVatTUqJDZaFLUVm5taNOTENC1tpfWkpQtFGVm6lUVcZCQ7i3xMBIS1dNUFFJNDiotbNeXGVMUde5t0tJYLVISVCZrVywTMW6TUtFRFRSUGJipURLYLm9r0hNSqyvVlRCPMDAQkZLvEE+WTtLTU+yYGVSSEzPv7eztqrjREWxuru4v79EXAW8TT46wsRITUTENsNRVFlBV0tZSEm2u67GrKhav8CzwMNEyGZYVVxXUb7ARrKlUrOiqE9hRlBRqMSzUKVFWq1DQKyjTU6zrkm0TsLFRUuwpKpCSsLBwUdKR0hXX0VLUJlXV1VVSLe9o1ZMtLM6QVC9R72mNEpDUVROYVBRW05DW1i8tUexW72+ulWuYPeqtFQ3VU20T14+PKZTsktGTbq7vau2tFi0tKxPRT9GT2BKSkpHXVnGxr9JTF9UWVKhoFJYYExLulKzRb1OTry5U0xKRaFMRUJTS05ESE1ZRkpPq1FRsbRMUqpSxlRLQEc9RkRHS64/S7RBVUejrqdfWktCS7dGw1xORjZIREBJT17Fxbq4wKxWt8LIvkSbpaM4Ok1OTE2ysrteZUdCSkq/vEdGU09WVVmmqa6/oFxVWkjAs8K3xlqwQ0tPS7JcXV5QWVBMU6FPTUZbQLZLRDq8qadMn02lRU20sLVTTk2RXbrEw75MTEy6umlbrLBfw1SuWEdOrcjGVDo3xFpMSaKsn6Nfv11LNatOQbVXrKyxtbCsUaK7ZK5KwK5HTVi7wVRUSUunrb25T0+wtFG1sljd2kpHP0W1u7a/s7+3HxpaVVhbYGNKSlBKS1BEYLO6vLGvV6iyTUZMu8FHwZnAxGCnnj9JWEtGsLdPS6mqpKzGyElJxcFLQsDBq7dJwL9PnqW1RUUFTEpPSFbDx7+kTFRQQUjFpLJOt1RRvMK9Tro7TL2gTF67WjymuLlTQVtHWVmst75jUVS5CamqWaNTVri4tadgAGifp1BXQ1VIWr5XrVtJom5GRqJis7lWVdFTs1JQR0toYFpGSr64u6u2tVtdqLi5uJ1WXVpauEtNZVi5TlRPTV9LusRlQra9s7Sxp1BVV6/CrbGus05DmlBPUbFZqEtYVq7oTVRNsltNSVypULu5XLTAvVyxuaOxsaGvt8KyEaJQUU9IXFq9vkS2tahUUmBgXVBRpreys1NRVUi5UFBKsVtSSVdcTkdcYFJIsrqmubK3q0lYsle4XqaoP16+ublSulK5tKtNSqtLU7ZiVKakoVDDrAW2TalJs0G0u8O0tapNUklJP1VQS8K+vLqgpqFARL2qtb1MS0Wnpru3sLBKSkpJrE9YTqZVYVW1XqKpqUVMOEFfVaFNV7S0tUukqK2zWFexr2eyvrKgR7O2uFNTsay5XbZduU66sVJeW1ecVqm/xLpIR71OUVlZXlWiPqhQrrG3YkM8XkpLTKS0tGFTZp+mv1ZHO7VIRlRqcGpTTldATp9Iomo1U1C7tKe2vba4urxhWbW7r0tqbmSwrrAZEkhEQD1JSlVMQUpNUlFYVFWjpaa2uk5NtS9Fo7RIQkxqWahUvD28S00/QxVTU6JdZEU/V9NvUki3ua6uVFTIxFRQwcAUuz9BXVU+SEkHo6NUR0q7tbu4TGRKQ0Jfv6qoP7dAYwezT7W3Rj/AubW9wD5PU02+rLGht7Ozv63IqKZoSEant7BGOVtmXF379EhDT1deUVynpFFVSLWyoEZQVGG8WF/DwFZcS0xQU7aytbWwSEldY1dERkNASK+3uqepVrq6tZen7/X3S762Yl9Js7a8tqi9SkFKvE1LU1ZMX7O4XazAt7RLSLXSZbW3t06vW01Zqk86TEmuS7C3uF5LTB4ipkVBQqNKTmhGTlivvb1SRl61rrKxqbBARMCysl6sr6diYbKrXEVNTkuxr7y0ULy4VLG5t1ytsL+gorO5ubWksEC1U1BGrrVPpFRVwKdGRUewvrFJtrC3OEm3s7xfrWBYola+TzZSR2pZrL2xu8S3slpHSby8xaW1ukA8RE2bp6K5wbaotrVQTLhdXbq2Sky+tUVMplKss6Km36lGXWWjqrC1s7im+Uyxr7KsT6ueXllR7vFVUWFPRkxiRk6sXjpSsquwYrpgwbhISVGttbaPrUg9RkhLRrq4u1ZOqa2vxKaqqLO1p8LGqb2+tJ+1r2JPZJxYQFxKtK29vFSpZV1KsFYSTKO7YV1PuK25tqNHR7i7VFWkq0JCU0tkXKBHT2Sqsbq2xcW7urC4r7K7KS2ksrNcW2JOTa2/R09IXkm5vKpi9qlXsr9JvbZBxKW0sF+nYkVFXF2RVrtMT6ytVlU9O7eyPERHTL29qUtOvrRHoaOlTEU2SEpSu6ZXvb5cSlNVu0bBnC9Nq7y2UUfATVC9QE9UqFjGWklVSU2vuqZGpKpYTlBkUVJD6qVYp1urrlFORWmkxJqkpk9rsFZQZUVMS2BDXKBIP2umtLG0si1YsEpPv72cZF67vk5MTK63uk9Xq7a3s5BZqLCyTLe0nKm3satJTFu0Q0n7QEdASUhPny8qp6/BsqpUW0GvWrmzt0KhVBiprE4LVVS2TlpJQ2JOs0lPVFRMSGFPTVZZXGmsvb5LQp5NQjpFWFxBSb9LTKO0sEdeo6yvokhIV1ZWUEtauE5MSkVCQ19AQT1FQUlFVFp4sTy0ULhSrVRNQ1FRwEtHQ0NMu0tFWbBET1dGT79jp1haeQK+TzlGr1eysURYw8BVVViytEZGSF5RTUxBS0SXoqFIPrWqtbxISLekoVFZtbBITb24prWyrGKqyKtVX1Wepaq5uLli9V6xPFJRTLNZVqW1sLVIUJtFRkRXwUhOQvqqOKJYqFxbwV5QS7IWLiuTXVlKSU9IULtOTlhYoFmXQaKvU7KxZr+9WElHUnOxT6SzpZxWT6e4RaxGRqoBmp3bsle6Waa+oZ60S2G8vatMT6+xRU1jW0xPTLKcnUdVW0+yr0hJQURMS7e0akxOFxilrbCppKeys61ASra3o0i3vq+dsqlRWj1HTEvBwaBRUqRlnLm9q08lUEy0tKaqrK5DRSwnRUSyQsS8NFDGuU1LqKKtsbs/RUpJS0kowbZXQkhQREy/nzFLr6+yxcNBukpFvL2/YrdBqf44NDK2S7HCrrxewsKfuLdQUqllWrBTWMG/Ua+hs6FjX7iuVLG3qshZvGPAqqLCvrKit7WxsU+zR7i3TU6kXllIRsW/w1NJS6mhr0RIT3euXVdSTru4pV62tc67t6CzvMVZu0tEv7e6oT5CW1A/UE69urphYq60sLO1qLysob66tbVUtqe7s2RLtUhGn7i+wqozNWC3uLFUTUVTDaO4uby6qKzGwEkuNKZQUa1fnlRRpMDBurK4rsRPTrq8uLW/vmG6vb6robjBT1BcTb9MP0+zUbO9qUJQSa9LRFC4R6/BTFi3sLuxtUpeZEaoYlRFT0lLv2NTUL5FQj5KT6u3vcG8wbGyvMPHvk1tXF++vkxXSkxMTkygokNMsLW8vk9HU0hUTz6uqbm9X2FdW1RMdUmir7i0vDk0uUk6rq21T1K2v6O/x7ytRjkvT7O2VV9MoUxkRzI2N7RJOjt1sbzBxry1tUq+vbe2tKxdx1xiPUpSrUVDnL68u2FIN6eyp2Wrw1lMEVm7vLZOpqJkq1i8rmO1pKcxtLRGRVu9x1NMUkyurb+9vFChprG+vV5aWbq+v8S9uaa7srm7wrxbWFCmp6i1u1TCSzI7ozlHTF+lV1VVS8ZPu75CQWS4tKKoo0JCWGCOVcK2tVpUsbXCy0FAxMIKQz9DNT1SSL5YZl9VVk4IuLqyuqhYRENdS0tXxEdFXklKS0tTUEbBvVHJQklUoVbCrhZdTki7v6fFsLBPTk9jVFNOOaVVpKaoqVJTSWtatJxcXE5ksbazrUdVt164XKLBxG9bt7lUU6+tvE5TvbqaXVu/uU1QSqa6vlRcs7a6UZioo6uhSru7ZKW4tbdOSF68vUhLREtEsqyuqFBOqKvJV1xcWbruaMPBv1KoYU5Ut1INT062VqhOSWFOU0VDWVpTTas8Q1FaXG6utsZOA6m5vLfBqqlDQsJISaSoqKlhW6mqqrzBXUuzUcNVtsC6VrC/vFe3wL6uo7vEV1BntVG1RLdTtrtKsEtbxalNSk9Qt0y6Ur5ITllTsblhtkxbZRHBUT5MulWftEFMwMNVUWK6scK7xK2zvEhHTU5dXllJRrexvbZHTMKqpUtSurrEwcDAnbmup2Cs+rFYWVRWWFi8uMJfI1u4ulROtq9aqrC6p6e3tF+5s72tvj5LR7evHaxOV1Jgv7U9Qk9WVVKVq2BJS0tOTba4u1ddqq1cx1SyU7S3psjBUklEVGW4UFpMXaZhUKa8wbe9v7FkoZkav1b6UqvCq6a3Tau2tqpRSlJNSEusrVNOsrWcnq1eXkpOT8DCwsa6vLnBu7zDNTCor7Cqq6i3uVO9SLi1okLAxLCepFlRUMVHwr7BxbdLUaerYL6+ql+RULq7vbG2VFRHQFRQREK1S8zKrlH9xbtkWl2psbhHTUxUwq0sxL5YSFNXvkjEo0JStUxPw8VGS7a+RMNZWKHHWzSbsa+yu6jHqK5ewb9fVVFGMKhfXqasr8HDT7CjC7BYXq+6VFJHqctOs1ZCXahEP6yrubdRVym5UFJSSkemYFpKR8a/wbC5uFRVsbq6upVbq6upUrW0Yly7sK1DSle2tsgbPURGsqy2qFNSXrTEtri/t0vXY7S2tFSrXAGurU0/sbWyUVVCSF5QVy4zVMG/x6VNUmS/uaxWR0dMTaNJSktHUlfHwLmusqVWTltfpFhRrLa4trW3SsC2TLe/UVRRSFxWVEpaV05EsrymuE1LWEysrLZNWl1UwFpGSExGvUu9trFLR7pKTrlYrF5bpETFSyBPv2RfrkW4xsG6tVu6v8PBw6tOucDIxkVlV11GPk1WSk1MS7ulo7u2rrG5wU9GrE5XUpurzqzDX1+yqkUZrb9j3FGzULCptayeo6y2qa+ytmG2sLuqSEqhWwewrKu2oblfRRxMSVdgWlGSU7fByLu7uUy/uqalqKulw7GqobK5pcG9VUFJTqFLrl5OYptYxFtMPalGRbJlqae3tFawr6O9rLE2VK5NRVbGxLezT0qrr8O+uUyosFi5vGDp5ktDRUG3u6+2uri6ERVVV1Cvqqi0u1VESzlQpU+9vaqntVxVsUNNvcFHwahQUaerY0NIW1NRPL25uK+vtrTHxVNYxsJMRL29q/89t7ZMXl6sTVA9RUdPRmE4w75VR01NPkbDnB1Or7e7vcG/SklCw8BLVllGWxOwvr2tRqXFoq6rwsC+yslLUaOvVaNXWsG9u7Fjta1gYExWSVNSrcGzs6hQWK1ESbBdtLmurru4tsnOR0SupK9MSLy/vE1FR6qprEVGS3BgXVtcULaxo1q1s766taS4sclDxEZCs6murLCwW1VCVVi4vlGvX7eyr0ioVlBWp0qzTExYSmC/vKRXtL68qL3AxK1OULC6uq6xt7pKUKNPUU9SYVrHv025tae0rV2qYa+qpbS5u0y1t79OTLjAU1tKR6VbVEdgWVRHTEpcTEK2WkhQUrBGX1lSRmBNRUizSLK6U1e+4rizWUyoXmBXYU5GUlJPwalSU8BERD1PS169vry8wLFYucDIvU1pYWi+wrWztLSztEejqURItrm7t0lJWUivsKqk5Ke/pLBWVmFbaUq8+1OvU7SwrVCmoKy1XFO1s6K2tbqhSVC8v/SzV1tJZE6oulRLTrOptK9wYbbDv7pNSbe1vlZYoK5gxlddrUxSqT5DrMK5ymO4uaKuoGBdw1VNTlVKS7qip6YOq6qzWGO9qa3F0LK0vqe+u1NSvritqcLCrUmrrGO9t7FUVkxNP0K1taq1tMC3GRuks7SlpqW3uU5JR8lBqFVGS1+qU1isS0BMv7tIQGXGzqKsoUBJWlBUr7m5uFJZVFDIybeyw8W2SUFGZLdPR75QZWRZRUxVS05USVpPRkioR1ZOSUZDWlFKSE5RRUREurvDvUpQKktGpi6uvLpawWJEYVeuR0VhU021sWWwS6WsqEtGuF+nW5ius01jNaq1YT6zuaezWZi9vWOpTFJTVU9US1NWPD2dpqs9QEdNTGlTT7CxYFBNqpmnpK6qvFlYXklPWGu4uatYtUWzvbvBsbOwY7a1SWdLoalVT1kxn1NVWFRfXk6nY05NSkqoSqy4u6O8TsLArk1HRVpOSqpTVmBVT0ez0WS2VlK7o6lARLi3tmdRUmClo1FTY7q+VVBVs0WyQElJT1tVT61eU0tmZFlMsbGmXk9VWUpNs1VGYFq2vmC5v7a7ULm6t7PAxlK9rU+qqVpZp+1Jube2tLAlW8G5Ski6s1m0tkhNRFuvTUpERLmorK6+vklWSEyztb9nYrm4S05MST41cEdQVKtZUlFPqq2joEeolLlkVUtXWLG0s7Wqq2JNq6qztWG5uLmpP1C+uayxsq6xqLCkRmBITVCoq6+XqFZCR0xGRE5MS19TXFWaQKKiq1FUXcK9q8O7U59Ot1pQYqBdRElFs665u62dn5tJWa5UUqRNZZtHTlpHT19LRKapubthW0lOs0Gfn2FPVLa2tbS7wbxMR2RQT0pOUUtdVk9eXl9KS0lLu0W7Zk64ua2YrK2ysb68S0g/v6FPVWhZYD07SMF/3kVNSqquqaVBQ7KyQEhGwMDCT0i7vUjWq6mxNjdHTU1Qs2OwvruuwLm0v767qL28uqepPDu8slFEy0K4t71FolJVSEyoRqI8qrBbPj+gsrdNUJ9bqqJWTkhDUE9e7F2ppbCmuayzW0Oqs6expGG9ukpeuLOyr89MtbGyvL9gqK6/uT9FRlxMUK+tpk5QoI+oW09VTq+sn6i2rJy9xaiusz0tz0lBra+wnlBIrV1Cl5tBRFHeYKquqa+mqtFcpLiytLtYt5/DxadSrkxKqkg8RFu3tV1IRk6yt75NUKCtUk2rpKU7REtGTaSurl+nYquwpb61R7ZNv0ZcsklCtV1KT61ZV0tfU1VHVVt3Wbmzrq69q023YalKQmZLQ03BS7+4W63F803ArkinX6qqc7tBT0hKSl8yVstZRkZRWKhLTklIRFSqQj05REucqp7Mxr2ttry3u0umpFRRsbFKR7rCZrazr1SrtqtBpF2gn/5VaU6dr8CvUE5LsE9XpKSuVU+ssqa5t72jvrpISVBOFalXXlytwUu2tbZUUkmUqEZDOkW5v7tOR5exolOOPaJaVk9RWj1BqMXLrnW2R6e1o5urQ6m3rrO3vU1YWlqpr7QfsKWzo1iztFi2tKo9P6ihSUhbVkNEs7RcWWBGRVn7/bq1vrdMSJ2sY0VKsrWosaypo6avsSxMTLq3prRIR1hYVKuvXLlJRkTDQ2qysqZYpcDCpU1MrEu5s6eqW1o+QUxQPz27RENCVfBPTEhTpZ5Pu734truzsWJJQEugSlFNtElCYE5GT6Ogw8a6ubpHQr+tXljGpU1ZTVGuTa3Lq68/xcFYST5JUmBBXV+vqcW+NbRaXaafqFNaRLG2pcJQTVy7n6zAuLRZUVPBwba3sUBIRUymYFpISb7Hv7G+tlFVV765spWqqLKoR7eyp2JTsr5HTlyxSsJXQEZLTlJUXLW3Y7HBq6+4uk69ZbG2tKanoktZosa+T1RaqqlKTVhQv1BKWr3GxKyxsEK/vLGvtMFQWVmrVFC2oqPKyU9SUmNXU2NXXFVVZk1Lv023S7y6Ubu7tVtRR11aTUpfV1NHuLuns7Ktqk9VsbK1Y6NSQ15JSU1MuktFsrBMxrlNtbtZZp+ioFXCVElKw0JTskm4v761uKHDu8LDxKquu8DDv0mbo6FCRLmrtrhQR05fY7y4T1O8vk5IplBYWFytuK7EWUOmp7FccU9TumezT1JSTUxgWVtOq6lMTqxMTEZbSrFUMre6rqa1XbNbv7S4s8Onur2Por/IxcFLSbq8tVlTqqigwqZfWbS6rUZCWkJDQqC7UrDBrZulwV5FvK+3wbWjpKYosVW3Vaa7Xa5RRbO8uqHCw7K1NzGoqsHDVFCspGK6uEGvqLW4wsO9vqi4s7i729dbV1hZXl9KUshJSVNGYLVITmKrrqlOs79Lw75FSEFARl+lqEJCXVNQU7lMU6yqT1bIx725x8tNRUlG/RpNSL9Rnp9VR03KsrGsvmNUQUpYSlBYukhBXE1PS7GySUPERr1Gv0FMrFnDYBmltrCvR6JDoqmqR0VkTElCTqesp1tKSkdJt12g3Jinp1FDt1VMWEGqTKhKUqFJQl6ntrMCBdJLukpMv7udqqi+vkVLRau9u62lrbq2uJtdVktISrWyG6Szsae5vaq1RkEMvERCUElQoE5QpLDBtLNSU0SmWq6ws0mpVEGrp0YjU1BYUFm8tKRZBbu6qlNDR1xPUKtNVmWyvsFIOKJLQ0NHWF1AQk23t6ZXUE6jolNapkhKTk1PsUavukNHTkNIRKdDR0BKSEg/uLWlsUO0VbVXr1FMS1VCQExBP0ivtbRJr7G0VVK0WrilXVxUoLDCUUxNrKixr8O4w8CyuVWxtT5JPVFURUE9Skiorqu7v76xur+2vEuopby8sLNDRbq7o69VWKKlvqdFq6yup1VXZERd6lWrR62wT1KoWaqzTElQTh5HRkJdwk22tOymq7CzqrGlvExMTQtjUFGcX1NAPU1KTblHSnJBpVeqQa9gpreyXD5Frr28S5+0tw8FGp5URaS3J7BJSKZkmqCzr67TU6m3o5+7Rli9vaxFSLa8u7NcWUtFULWbnk5TTa7GxEZMRkNLQ6eyZEREFhtYVVKnoaCwuFQ9Q7izp09KSFucVLSrtUNHR0m7Q2xOSZ9dGb3DpUtBrUG2r6yrpapCPlBSPj2wPUNGWRIhTUZJpalQuLhOSUdLT0xCQEmoPlNJQ0RDXUxJS7ez"},"fc1_b":{"shape":[32],"scale":0.0057624,"b64":"M0xPemZuenlGLDU5c2p7cXRobnRnf3lhUFJveHZjU1E="},"fc2_w":{"shape":[30,32],"scale":0.0079344,"b64":"nGOmoKKlSTSesWejV1BLKJOfWaM8WZojlpleWFdeTzWVT7cxqk6hNqyvoZ5HMq5VpESTqDmhUUdRX0tIl1xMP5VbYaNpY6ynrZuWradkYa2aX02jYWCvZ5ifXFypsJRcrqmdTZ5KX6teWaidn5k8NqdCRUVQSFWgTqNPQgenTZmeq66mVlVPU7FQoKROWUBbUpWdTalJpVmhraKiUFeaV6ilYklJUFOXVl5KpklRTqZaP06hUZijplakTKldmqegrqKvTURNpJuuaMxjpFZLRVQcXSSbOaGkU5+iZ6BdU2ebpHaxpK5oa5yVZZltsrSzbKugY2S1bqymobGiXq9xqFVYqaJUmklQt6ZFYEhESVBWo1CnTadgRp+kp6ainVJaYZ6Rq6psrp+upWNoZ6Ssq6ulXpuldXVZa25ir62jbqCxrq5YMzwuS1G5rkJApZypQ0lIT62fTEOwUqqvUEOkU0OnV0pfOKOfXbFYUZxbXZurn1NUSlWaSJ9SS5ybuFutWJ+crKOmnGNqsLicpJduWKubV5VimFlZoGJrXGSloWhZpWmhqa9noC2mn1u0nZ1YrGmdVFFUSVqmXJlZVqJeWqecpndwo6OnrKGZkZ1hY2ZtaKhdtXbOq6W8sbNfpJaelKCPY2Bja1eMyaFjV55moWZSWpe5aFSNmoqfqo9pmp1anNVdpbdwo7JnVsVZmZigVWSqrmhm6qRlsZhuqGCpQ2eicmqmdb6mqMZ5bcSgEKunv7ppr6mdqGFxXnO+rKql/TlZjVtdS0NcXVWdOaOeRUw4pa86TJrNMmVLmKlKrEpqlbqYpWG3T6li0mauZrCyqVFnpWtQpqVapVizb6lWYudIMF1Zl2BNWGYXVWJkpqOnZqyvWKSaaQlXmZmnWGatWVNQSVlYm5NgaU1JmFCap3WSmJ9ajFpLRjlSq6VpLahcZk+bm0piTXXZW2ypop6npZmlZKpYaKOibKBqoXRmp1VcXqmjtKNFXFnKxKxaxKhlp69wt1+VUrBlVrhiolZIoFxsmKChjppkZGhqkJVvYm+KqVpwn6+YXlCfa5eLorldO1Wgq7GlY1BZUDNdr0YtFVy1TkampZ1LmJ5IYGqkqHFxmqCmqn92nLGpn559pG+0n6mvpaSyq32ZqHqrrpuZYF9cS7WkoWOacWFwtKCZU6NWr6erX06lmWVhoGK2vTikUpynrqd4oGdcWaWoq2aVZ56wamOconRXcZtfom2xsEdPmJ5Xp1mkY19lU6dbpKisQVGwlqFGXjUsTWJkq6Jo"},"fc2_b":{"shape":[30],"scale":0.0062732,"b64":"7Q8g+BOl6nCsXD7dLsZ2u9gIgcfHtsHC9bwF39m4"}},"decoder":{"wte":{"shape":[307,48],"scale":0.0346747,"b64":"CdH60SD15tsV8voW2www0w4IAfcE/Bv4Lz0C5ekZ++39KubCHwTt+PIVGRQGGDkDKxPkK+AiCxgIBjcK/igCIuXp+gPZDtk/8iz3HgPuGQCh3wEJ5RkGCgPo3OPX3/wGGfwQ3wf/7QRU7vvr8CbfL/vtCe0tDBEcHAkWv/Ci1PvXARL/+NsJFO4nBRLu+QsFCNHoOAcT6A3OQN8RKjMt/h8zJvDzBAPRDQH54gwd/g4VOLntz/UkQvkF8d3eCfC7A/3u5Ace/vMABbwFMBfg8PjhESDtNQMO+Q7n9OMA+BjcEggJ9bYPHdvk9/ojB/wSFCIF2uMJ4Q8F5+YD0R8uTgHb+Os05R0lDPH57f0F/gcUAecB3gkv/gMORuvq//MEJevtEBMaNv8N9g8P8gIO3ToWCwsE3+0uCPXPEh7pAA/+vTwKCA4C2cUODCcNGO33/A75784DCyv4Dvr15eoTGCA+3Ag7LgYEKxexLUAA/9UezQT2DSEAH+T96iTvKAkP+SIf6NMz0T0K5NjYF/QIFfbcyRvoMDvyERMBGBnKBPLY8iD3//nq7QYF/w389C8X/gMUDAILCNTbDQX2HRT65/D+8PDYK/T14VjvCU/r/v8I/Q4K7vzk4f7tMTvjAQm0IiUR1ggY6tgh4R0v/ffmFjDo+fDb4dpO9PMqIxZA9PQsIwbmLgPl4+IeyAsVsxAU5wwT+d703tILGBwIFt3vnhkLDy0MEBgD8wITBwL/E/IU/goBBggH4fPw9BlRzAMcKvANIvUAKvDK4Ord8xQREuzP7wHuMO0D9R8W4yAb+Cb33tfr+hP3+OUIKgQc3QslBfkNARv8MdzlMhHNFfPrCwQoCQ4NNtoDCuUG6AsDTQ3vGQkBBPQG8Or0O9cPFgjwGOkx2son7xoF/wsVCOj5MSkN9fQSCRYRFhsP/NsU7AL4zdgFJjHuEBAGNu0a8yHvEtz9CPkh9PQIAyT82cAC4Avv++n44Ag96ysWDhMBve/t/RID9yPlID38OvDa7vb2/SIDHf7iD+j3/+8L8jIE5Qoj5t8XrfEO2f4o9hPVFkPA8jDZ3xomQxogsPP09vj/AAPRxAIUDhcOJx0BBsIO7vsTCO/sHvccBC7z8yLk7ebVIfT1zvE9MQ//3hfPK9QACAEF6wD8GADtE/keFQ8T7/cq9unq4PsKEe3+2ccGK/YdJgw1IPQQJw7q8zIL6Qv629oBJgoSEf0OBSsHARMn7fzKUTDIFvsr8RUkzsrWAw7K9vj25xASKSQGDbIcLjj85/EI9PIAJ+4B+S4cBPYAy+v43hDSCCr1MgYg2AwWE/vpvAXcAeDdCvYUPjTqESXZHOgq5igUyAv1EDAe3vjz9e/Yzh0cO/Du/gEU8uoLHwz84R/wFfb/18UE+wfx/xPt/f3dINzLBPEJKffw5t8C+gPfzf8bHf7qDwztGxgqwC723zErBBkHDNgG9SgvthjjGhnlIQQpFu8IKj3YCwIa5w/+2QAmIEMDAvoLCvYPAOruLiHz7hPw/Q8S5dA24/Ht2xbj8PM3FwEt8Ak9Iyja+AX4ESb2CA746/beEcLP2DUN9s4JKegQ6NYN9B4s2RLKByj81EwAHCEBEBIY3u3wE+/h8Avb+/7k9AsG2vEKBBPlCQoK+vf90/fkEebmzCXv/R0REfjW40brDOMLK18K/wDvHwTe9w3U//EM3SkBKiPM7wAIL/gR++MDASgH+tUc3Q8tJvT3LPDnDeIdChISNjf2LMkFCBv9FgYyEujQ9vAK5NvvAijo4hzxHAIW8fsV/VkDDNH+KuwNFQwBARbd6NH7ESoUH/wJKy45693a1+Ib7AoeEvE069YjAuvvDS1PDxMT7t75FtwP8gMdytgh9OzXFh/3AAP1+vwBGO0B2+sWIiD8B/4n18cR8Ov080sABBU5AM4Q7A/78gPaDwr63uYI49va/R3lHujqMgoY7yk4/sQiJuP+3TD/69fQEBgc5vX9A+O7/jf6zdAHJQzBE9HwJA/6BgzoBS8G+fwIDQXlD+vPCA4bEdwVBBcSINbz19/mEuzoCfPo1iPdKM3IAuvz4hVAH/EC+BALyicpECb7/ADi7SI0HBIs2OHzHg7nGrkCFP8f1Q/e6/gb+7wJKP0k4P8RHeou9xErLP4LDBIR5iwA6Q/zCgsSCRTvFjz10RMw3RTk69jhMt067/jo/A4aKAcIG9LzMu/qETMQ8goC9RTp9//kAS8dIf7jDjsR9hYG6BPlIzEuCTjg6wb7XAsHK8ggGwfICCbhNBwPKBUa/CgPRefo4AT1tg/ICC4jEgcCBiL/E/kHPRoarwMOMxUr/dwXAQTfAevxCzMv/ifWrgDrEOwQ4gLpzt7Z9QcC5PYJ7wb5Aw/a8B3YAyb1GSkZ0/UQPNLn4dq+FyDeDAIDHd8M7wz4z9gH094aJODv8hDb4w/YGOsd4C7yHbkY8g4K3gIM/Q315QwQ3wvmOhT+BwkX4QzvGyQ+8BsHASUdC/3X+hsn8M7x/NET5f4Z7AP+7PP5IO/R8gzRDhfABar37BEHAREIM0AV5gD/DfIDA/zm8BMUQUfPIBDs/gsN8uTpBNbqHy0pI+XjFdsDBv/o7N/pNAQOAiX+7RcKDkj55c4QGBMAF9ngCefRDy3UUd8h7v3X/TMHAOwwB+8c9/oKH907BBXG19YYPfcAIRv3BOPjThTx9uH+DQ63C/co2Ar82gPHCujU+u4aM/AiHfMl4Orw1vb9GOn34AX4+QToDBoxDtrhuC4L1eLrxuTo9xYlK/zo7SsL7xALvCHt4BjiEu30Qd8MBgIKKBsa+M0P+C7vDAQvGxUXMAcjINX4NQbm/hQHzuHxCx0J9SX48DHs9x7jOv4jDS8SIEQKUgIC17UPuOrs8QUR5Ng67Oz38g438vYJLtvx5xEBKsnqLfbdFdcgFiJM8xkAG/njBxX04dsGNvcZ+woFKCUt+vreCBc78tr78vXyEBD6+xvPBQbLGiUTC8Xp4u8YsdMH6B0QGvkmG/rzIhPs9OPpFPwWMAH76u/zHg7/1jQ0zNP87xwXCuvhF8oB4NMODgzgOfhAKxBKCxPlJO4gGjAYEyjrBhX9++Ag8ATo/iT1BPrmEQID6SbY9gACGxUy9UwB9AHu6RbIy/EqLwDG5gA4APgb+BEB4N0eFvRC9sIDBTnd2s4P1P0b7N0UPgoYIvYeHPkoD+kJ8Br74hjvvL0VF/P4MCHe7wvW/vD8F50m+hVEBf8A5eL8E+vfvyAQBxkO6BP45Art2RvsBvUFDwjdwvLWBM0V6gP+4qoduv/iHy/o5QfvJ/ntAQ0G/EH0Jd4U+e/UIAbnC/US+yH2Fv0W8+QfNu3+0/weDzgMJrzrDPvoOeHw2BMS/Njn+wvz3NkgBhcp6r/p9di+IkDdL9Dv20InEwfUIxYt9BPmCQv6/ckI1MDvHAL5CzAI/+i+zy7z+P/h3dwzAwoTsg/+3tr/ydYeHcUbBgcBGeMq/cre0esZUugh3xkD3+vsBCQNAg74KfI0+QQuEvXtDscN4QIc+u70+CQKSgvWCh/j4wnfBdn87QHw6dX/+PEEECnuCvHsCO4WCuQHIwkk+tIdvikVBy8VBvQA7coKHhXLDw8cJBTFEwf//hwQChsAAAPr4RoWGtr7DeTF780d7/Y0JOrhEAQNxesL+Rzt9hxDDQQaEBgxHP8D2PrH/On8Bfj38cod4wqmT9sT3kYbEOPvTPkT8f8jFQncCCrsGh7hAPIFDbsV1+jlB9Mq6g8SCPwSBgdE8+0DEw/iFtkF3+/P5CzpGDC/9fsLH8sN/BXrAer96yUHHxfy+szSCAkCHwTy+98iIu3i9RXepPLgLCj20f30FMQD5g4PGfPz9vX68Ak23xTbDwD47eEd/QYOId8c8/+/5RMP7MssBA8pMPruKv8+/CTD+gX51xIU7PMG//3d8AENDQ0J7fD7ANr/9yX85RUWFhQhKBDp5hv08Lr+3S4E7grs5Cn70QoUNg5KMuj+9evtIfrpE/zVJ9AVzgDr1+4tzOnuINfl8Or32gwQ+vT1/BIbDgjNG9YIuQcl3/v89rbyxCULq+4T4wb7/9k0+vbrAO8e9PbeChDZ5BI74zn2KDcX+vIk3+/8C+kOLuTp7xqxCwYOA8v49+796kD9BRLyFfcJD/sK+v0a2hrtBTIBHizN/yzf0r0LyuHnD+kEBw4fFDoBDiT8FDDnLMILCy0pMvAZ8w4LBAUk5sYCIg0H3vzl1eEeLyAwHuXk1s/t0vjpFjkh0gQBHg8W6gAOA9If++oKFv0iGRLx/QUoMhjj7tPkISHmv+7qEiEj4yYbAhjuxggY3yXuD/D19fUsMBo11wQX7zK9BRTsDCDW2gLV4fcWHcD95vby9QAu7vjp3yAvBQUIIOsn+AUJEAPmx/bgySnXQB3M3SAO7T466eYv4/0N9+Ie/ekOCDzW0BM/Ac/86OPcBRkXH/b+4gkK/xL7MusjHRXY6eb0ABcj8wHUDeP8Bf346wXSGOH1Bu0qIRTs7xIxGEXXF/Y2BtLm6kXu8+r9Dvzt3gn2vwrfBgnI1gnE7eoXAxDt2fT5+g0x9QEWG9D24xjbKxDw+CAvCdMpPwHQCP4EBdnxDBLbxQof8e318CIFBCLhIScc/yjT/tc6HgUVAP//JtsURPsWJcrS8swW7yfgzhNGB+v48BkV5wQg4RcCDgQV7R3yCPXsH/waAy0YBSUfBOEp0v4t5u/bCSc7DDIBJP32EBoF77/v/jX39C/7AAcO1t708QPbCSTB6fPs+P4W/fb44h0/Awv28y35IT33/+YHPMMuEQfvJN3/HvkDAfcS59Tf3foVC7fgCAr0GN3MBfTfsewJIQEQMQD/I/Pg+ib+4NTwCAchIh/jBAjaDs357fEyIv4HxR8N5P3jPgvRHQu/3QYEEP8GBxgt9BMF1CIS+PErFOIL9cQYB/YGJhwjSALjMg/A6dr/AgnkAuMA9touEA3r1FoK9/IK8R7FxSQHUgId87X18BK/1wA5MQ8a/wwh5B/o77n7HdzsC/EWF/v9+Q4eHiAd6BEa/+ACBuoj9K/n5+w40/v7DPP38PwWEwPrDOACrPUOBgUT8wjxAMMSBgvoIPn75N78DCXdAvsE/xLa7ezk9tXJ4Cz37zQABOrmxhksFegF388ICM/sD9X0ANpJ9ALf4/PwFfQd9gDqK/3nBcIV6fL3GcsZA0fsOgzNPO75JCX47+QLEP3xBfwI/+3dEQUO3AoeLgUN7wkhF+vzJNIhswnDIu711xj6JgjkEesj6+nLvhfuGeHP6BPWEPcSAAkrSRewKUES6C0Iz8wOIgAY8CAPDREA9NoC+Cvex/vqJPIkFfns9xEN4Pr76Cuw+CUL3+P9+gwLAgfj2+kZAO726xUWOgj9BvYExRQ1+vMVBfXLDu47CwckK+Pi2jE3E+4y4fz+6PsG3w/VCR4X9co7Cx/wBA4gBAXZAvz4xQrm1+n9CPbbA94L/N7t0B/uJB8QG+Pm8gwKT+LjHxnrBQcpAgYu1iMqExz9Af4d+uvy/BXlHfsO9RvoDSkbBeoN8g389hnw5O8BE9v2Phgy1QrdCwMdCggEx/si5ejsDvcm4tQN+g/+xvUCVgn6OxXYBSgp8iDgKvHJG9bw6RJT3w314w4B9Bve9hIp8CsT9bYYKdEA4M0OIxr84OUNBv8q9QQXAe7u++/1FgP8EQsg4gQWDhPfFijH9uIXNuoKABn63ifY5eQW/CEeKR0lKN7s7QApLyQf6cLg9AcH3eDWCe/gAAf/+QQG/QP0+fsK8j9IH+7v/ybMGzz+7BnszQg599/i7iUTLkMyHBAcBOkHCvYVLwfh7v8dPTUU1/An7RD1z/P2E+j+Et74/PDWGgoAHtMqGRL6z+QT///71NH5Cer40fPqAAMO/froHgj3+xz/FvHD+P/5/hUgLu6uCfEnJ+/a6fkT+xPp1BkVJBT52hzQEvPp/xkn19vYCgYfHdoYBQX5zgb+/iDl9h0M6foB+BYH6wz1ERMK9w5U+zP+B/fC8RDp1AMFJA0eBe7S5uYm5ibR5voB8C0U4BoC3MYKDiDZPiLP3egD2v3x4O8M9/cOMeveAekHBgMWPfT+UvH65Q8yAt8aFOfkIzQP+AANIAz+9g3+5yTD2enJ/RvnByUJKPsaF+05+O/AISss89v9CCL4+grW6fQF97ww5OQPIu4MIw3+4iUp4RAgBBrk3O4dMPX5AgAdGuL49QYjAP3Z/sYh/CwUF//k6NIVFPwA/tkIGggD+RnyEAsOHQdl/dgC9t3y/B4KDtkB9eQGMcvg9DfVEPECKAjtFlDe/vkmFOMh1bL86A0LAwrtFjo0++AjCBTLH+n4DgvtzvIy8C8X7OPANgIWJyoV+RoT+Br0+SXs+hvO6PoM8QclCBr8IPH43wu34Mi5/xHd+f4eCBvl/OowBv/y8QEEuPMM7y74F+ovCPsK+hsC+TYZAf9LHyb98CLkKsL68QkA9iDcIygF/9U1C/IpAvXl880e3hcV8Rjx5+n9JgsRAw338uABDsfv4f8r+wYZUPfm3w0IJPgfAyIZEBEUId3PKf3o1wvtEAsQ8cze0QAH5A7L8d4V7akyEQAaBc3w5CLm5jYRAjDv5N4IFRYV7+EHNPDrv+4Q8/MuDdP9C9flLQYi3N//BvwGDRz1FCop4BEM8Qr4CCwWB+/k9NTv8NL99O8KEOzk+gMhANTv/iz67Ovk6eAR7x7L2uP7C+NEDScC3i/2+0LkK9Gx/yAOCiLmC/fH5x/4EgE/EwPv+ufQ2Dn4GfYcIdYjCh7/CPXv+Bb9ERg4Gh/y/woC6xgJ/OInxCLuNunz0hsp8/kOC+o4Iysl/QNO3BYcIDrTAOnxDvn9JgPy9A0j8+TtQPb/DPvnAzUSFO3+1jkX/uPg9QXu2DXzARHtBQEGNSH55/gF2f8RGf7x6hQKNvL2Jx0i7BMaAkAsBA8VBe3yOyDL9SUC8gfr2O0KDAgCEAnOBCHezC0S3N3Tv+vd6i4d8wcc0CTx5/Du0g0b2/YUGRcQ+/P1oRny9OEW8tgd0Nsq5Rwl9ADvGvfoKzPt8O3rAPno+y0AGg8VAStHBvzWNisB/tn58ecR+QknIvYaDCnuHQT4CwYS7woS0QvczP4gCe0N/gjQ/OHqFebm5L/s6CoPKeEJGCEb5y0AEUAf+Qos3wgI1CERJvIU/+Hh3tQIJA3hBvz+B/gGDtEB3wkm/AMO/xvqJT8X+D4+9jQK/vTyICfaM+Px+OnSFO09ARn4CTPzG/34N6wS7hXlJeT7IyvaG939JtD68v/xEBTlItDXFR327ujm0/48K+7ZCODZFDEHCfX15+/66/7dIhv88AAG9wYY3SsSLeouLL7uBNfmFTEC1uMAPBTsGx0Q7eb01AbqvPIgFNfiD+4a3Skf0OQlHhkoARfr9Ncv5/cf+Ovc/UEPHwDjGOz2GxP/HQwQDizTJ6sOGjYT7xf1/BKyCP0J6AcB3jXcExcGDtQcCzMrEAoHGwvqH//JHdcGD9n+0gwQ0/bA+A4QFA9BCAsaBBfdLu4yCRoP+Q8K9BT/AMr00cH+ICjh2hP34c0iIx3f0twa4xTzJ+0AHsnxDQsaFP3eGBP/HsL76CAG/hDc2gEQ8fLgJf/l7UDj+gwTATP0/xXw2Rjf7j0VASby6t0Y/wMZAe0EvSoz+SDjBSsVB/gG+gYgGewp/+MADdrwAgSxAB8V4QoH8eMb/8wjCwYApvUC2hDmJxfy//0p8/3z4ygj8hPqIfEd4PABJeoI7jn4CAUtEyDeJ9QcGCb/ByXzEwr4FfMX+AL7CuUDzB3pBNn6+hE+5R00GzfoLC/gHQn62RklBgcd2gom+yfWSeAm9//x+fPGAQvHIiNH7vwiPtDvCOw+C9v72r4pKhrxCNvZ3NMV+PUIAdsD7xHfJhDfDgTZFQov9eMVyQ0T8dr/A/4U5RgJ69gP5vUP/AfvB/f8AR7NDBD3I/r1HywHHSIQ5AMY+M7sIhQzLesxEeX/B/hJ5wvR8w8RONYJ1Qcq8ynzFtkZ6OsO/SD8DfbyMvf/4gX9M9wBEhAFEj0HCfi2ASf63SHs6gMWKc3+8BPD5+kJ6+ZAMDbw7gH25RHlEPg67+IcEOL1Cx4AChoM0cwE5uwnAOr2EPcG6u4/Ej//FxYaG/YoA735FQYFDlMXESkS+A0W9gTY3vkLzAsLPQUFG9LV+uAXJOgo4O/2LO/uHdnsyif7FRbo/u/4Evwp7ODvAxMKC+LWAOLWOdwiCg4Nz/Py9R384QQ27hMIPc3g9A0dFxgWKBLpEigA+dv37iYE+frertb30Ar9zeHz6vsf/A8YAecKH/E/4yOy9t4VB/QVWAftFuPd+wAKRwH4JefgCN3wExfl6BIuI/os383MC/olyhrd6+7rBg/RrzDw6RHuG9j2DQPSHt7O7DAR0Bvf+encIxHh5//vtdfM9iQHIwf0ESHNwgQCFdz0+vMLAQ3mFyHW+/IfNOjXDQUQuiLSoBHe7wkU/OzROu4Sywnp7tj8BSAHQPc7Ahf/D7vhHCcH7BYR3jEPORvIAgQqH1YtMgvzFvMmACcD0/AH8h7OBBXoGvndx/cb7vXXBeUdQ97Z4BTWFh/fGwzy9vIxGfgS++zzJAxKN8UrSPALDwQfRgj/+1IUxeMG+xn9KOH09TkaxQMnI+7NDQ4GAtodAfQiKCO+/OQWBvbtQQrrAd8BAhLQAtgKKePhPQInCas6IREeQP8KCQjtDNwA7CMHyeYXGxgDGOgM9BsiD/7Z6iIQGOkSJCbr0BXMCPgiFzzf+hXhAfIk2Q8CGkTMDOr4/+Ur8sonDf3469Hs1BT22d/GFgn6+PPvN+QQKxXp698o8RL6LgMSG/gW9PT05/YOPecw9vLwHf75AtkM6f3t7OzrBt3iEyf7Hff/6fUHCAQIT1YLCjXP9gHvz/rf9RELCNIH5gEoGvkQaM8CDNcB59HoQrntAunU8eQt7xEoFuw08QnpyAwg3QkLK+PH0R0TDjL2Bg0TGP4IA9gOABoZ/+Hk/d0LBBXz4P0QJOgN9hcL+iAuH+sWDCHR1Q4CxQkoAb35DAPyBfkbLfP19/I78g/9/iYQBBj+BRzpF7PeG/UF7xblOCoStwDh6w0P+/6zEP4V+OPMLPfh7xDSCRbTARbN49vXHRL5H/H1Ixv48NrSKhUY8RYX6ywTCrwL7xb7BN/7ABnNLRYPFBoH30Ak0wMhAwP3DSnz6dbE4QX0zED9HB4fDiQGIfwsHgj0GT4L6NUh/N4AAfwm3P/3AwLwJNgsGsI3C8PG5dMWAB33+xDm1x4h+RUx4w0rAwD/Ef9GA/EyGh7w9frm4Tz9/vgJ8P0N1vf99BT/ICTo5eoA+AD/wcTz3FroGfkf6hD++vj51xQ80crw+x0pGPcEAzwbMQ/h2Pbp/xMZyi8tsiBC/SUQ3/Q6AiUbHSTZEPy8MNv/3/kgyObx4iAI7hn7+dv17f3sOfblEOj3AO4MBu0gysEcAzLs4O0IK9UFF/oh7TYB3SL/8f77FzIz7/8AC+0CFjfIBOsxEBu0BQbR+ADmGNYdB+nVTiD/9hsUAg/s2x88FvwfAQUSCuDj3gzl0v8U6AcdAcz67/Es1M7vECMU7s/x+Mf2KQEJLAcaABj5HuL46Rom7wHX1e/2+usMGccq5gNA7P3x+SrEC+jH7RvnDwMQFwnP6frj6gIH0OhPLvIM/QwL1wrhDOAP7s71DNzbvvULBwcIBf8tA9oY9f0h/fMZDuwFCgomAggWLg7rHfW/CkfuJh4h2OrmEALsBgH69LkGzQvo/er42/rMLMsYUQPW97zhGyExANPw6zD+xwjNJPEWC+ADAxbP7wri+xIoNCYO5QDmCCT/IBvVEOzb7Osi2Nzo3N0cDb1G+xkt9AbG8sr83xgb///hDAMP1AEGLd/8z/wKBegJU0z59BH8Fu4e5/vi7QLc/P3P8P/eF/Q6APgnCcwT+fvt9gzkAd/a+hTmCxfxDfTw4dfd5ywNAOfp5BIADssD+Pkb6QDKAgfqGRwgGQAZGfU+FQ0hE/YaJg/P8yEOEcfa3hYHLxz0JhTmwkUBCvDl8M7r/SjJ/fbe6Onj6yXYL/4X7Q7+8cTnCQYFA9EV0PLzA/3/JBcgK+P0KCP46wfsJRjmBP77Iuzw8gTtA+ks6+4b/+Qc/Ov83BnNu8wT/C/XEgr/6woiH7oaKBEd9d7zCyIyJgEazswKKL0B7748DAq5BQQCDAkQ7A/RJA0cxwEK+wUcEu4FEAgAyTb1BBfyzAHd5wYT5PZTJSYNNwXi5Qf18iPbvu/bEeEW4ukXKfPRGucA/wEUJhre5+zjB+f6MRIwxf/0EODjAfEMDQ/dMCgKFg/5F+ADHf0DEQnQC+rx/gYF8/cvERs3EQTXNPsl2Bf+G/cX8+wn+P3T9CoDz/HeHC74BOE5A+rgHf4GBwPKFRno9AsFHkYi4gkzCAW2DiQTANn2+Rzp2Orv3vEEyjXv/9ns4/Yi9fsEDBcDS/caI+4jAP0v+gL1HQsdChQYCAzt8/v/AyweDETp5/YX/RnI6xgh+eoCEeTqPigu+wgLBSLo6ec8FDYdSw4kx+3mIhf5/fITBR0UBfBI9PgM9N7dv/L+3D7PANPF+A3M6P0RAAnY7Ewh6f7p5+nYEvEn+O7UIdgRyfb03RQvyhfyBPsS+eQcHiD++Bn5NdcVNhslz+T1ASbuHQEW8drk9jceAQkXJv0aCBz4EuzW7eHcAvsD8h37DPUsK98N7hUG2gon4uv+Qgw+Le7hHv/z/fsMHPEN7tXuD98AKwXo8in84v/QA/sbAf0w7/oCCgst2+Aa1/UGEv3iBu3kIMz2BwWbPPcEMO4CCvf96T7nIB/m8yTdCVUj8gHK1QDx5O/fw//uDdH5EiT8FgUcGR8u/hHCzAAb3AL34tfPKToI7PQHxgUJL+vz9dj+Dg8MNv8IJv4BvvMuOw4VCD4DGgfgB9j3C6fstNUK3RsuCjDyFBj+9R/o1Rf28Nru9QLVCwLlM+nf+A0EBO7vHBv38PzfJ+UGJfQEDBf3JB8FCzQaB/n4w+n8C/wCFska8T/QLuwY6mrp6SkQ7//5JPgBHAzj6hIINT0w6SgN/i/a/Oss/vruCcIY7//t2RQxBDD7BgXaKAYADB3v0BQi3BXk6BrrAOf2PBgD4/kA8hoRMSgJBiQj6SH1zxvX9Bnf+9QKCirSyfz19Qr65NQp9hq3DNzyBtQq/gwl8B0T+/EhIdgn/tYEEuzz9e/tERnrtwUn9APlMBr7KMIGHjjaAgbpAv0SBfAo2yf4ECzv6NsZ8xX4Khje8yYVBwvlOPsJ5uzLAwUIGxksMzb1GP4B5vpAFfvq5vr3FeAM99Qj/+/Q/xMGJtce6wIN6t/3CBH9/O4GARD+5+bmCgfyIh4QAe7pCwTz7f4e203A+AXk7OEHFdXG2Qzu3AwWzCcY/9Hq6gNBAhf9/fP46QoB4fPeNw4fEyD3KeXXKuLkEx8vOS/j6B7nQgzp3fMLGiHtABbcxAgi3xvzwjXj5vgX4PIPGhUe5honUg/wG9zu+vQN5/7XBDHY3zEvB1YqLfRM0ugI4RsyBvMzvxsp/eIdxxLo4PPNQ+TnwCzyGNzxBCb56PoADQkY6if89Pf/Fe3B7gD/L+4VIQUl0vTc5QHeGeXnFh/1CfkCCVgpD8nRF/0QBRHRBfwc8Pjx8tsJzyodDe/48ege3QECAR3K8grizQ1H6wz72xUgEPQ+5hr9NxcVKQwp4BrpKdoBEsXo1u05MOv5CRfo/BH5DS4R8gXb7zzv+er1EQch+eolDjMfHAYHESoYMu71460kEiHs8xcQ8TE7DBTZGUzZADW3BgD75zwlEAUV9SL7vBQX4hy3FxMD+DPFEvoB/t5DFBvx7e4l6hoiP/Xo+OsRGvryxPX1494KMQkM4i3f5g/MJgUcIQX2HQr1/Bf52BO8KxjnNvL2KgH+KtIJt9onDv73FOPR9wAI4RHrAfQJEe8p6CXcANYN9DgJ+P37+AXj8hvrCeIBBRtLJB0cDP8MB+Ed/gHH1hrREtLv8/EGEz32JxkB4N8XEwYQ3hsHAuoM4TII+r7vEda1+vrk/TQpPyEIThAREeL1IwoPG/Lv6loVCPLMsAr37U/oCOHZ8QsU9Csk7PC8Axga6P743vPy3QHsDu71z/3e/fP1FP0MBxwyHQv1NRsgH/T15hD58BQN/eUD/hU/89sD9/4b6i7l5gP3HBzr8RQ3Mv0mNvWqAP/Y4CDx6wonFzEURhTi/jEI0sn8/A0l+QD3Bx3f/+8dLQu2/eYO8Ovs2T8PI+gFKM4H5Bf3yfAxLQwFJBvhBC/t1hMbGPkLzvn6NvQfHsP24CgxA9ntEtjxBujiHCYs+MXcJQkc+9/y2/0GKwsbLhMUATQGAhQD6hYuBQ4mDNxQBiYgK/LeJhIPFS386P8Y6xvqAwjh+TYFI+Md8ggZ7PdDECBD8CQDDuLv0fwu7xjWBMYQze3wDDPI2QHaLw3+4g2xFfb3QhGyKiQ9GPwi0Tj+PwHmLR0C3AsR5gi69APy7yrlUg4+HQQF99rWEwIGriL16Rja/cAO4vL78AUP9Bvz8u0mzPviEP5fCQXn+fMbDNk12Sz7DhYV1eDr0ggl5fbTGe7v3g8jyd/+EeQAJxAC8vDtM8TUz/X88CkR3TQd+dLmB+z8CSnmFN0T2OsX7d8CGwoEz/kI5RnSyyfRNvfwGPa8GvH0Bjn75gUQ+CUfK+8/Fz78AfwO5Bj8As3kCOsV9c/Y6xonAtsK9+z9GPoR9eMD8uQBCxv96R/7wQ8gOD/0DPLyDt0WEzY9/C4JvhsdG+3gDizd2AEiC/ABLtr+AO8aKyAH3OvzArfz9vsAGgsLHEfA3NIH9ujDDhIXESMMBestKuYkCBNB9h8KKc3zESED6wr8+xMC7gvQKeEeChAJG9zz8wD+2gcg7k8DDwT/CjgCCffnFN/1AQgm6Q7x+g/jBgY6/78LEw7uAvHdCPwMGOzX/ur9GhcE5t0hFBMCMMvV6P7Y9xEBFSr/AeICEiw9BCoF2ewFCA0bLQnE+yAT2R3mB+zFDCMXHhr2JyMUBhgcJQDe1L3mFhgM+vYQCiD9Jh4fANTJGDEr/wD6HRwIKxQC+Dj3AgElD+7pBAsXHBQqxtnjHjoIzRrZwvTwBwL8BwIK8S7XCRv+9uAJxxDbHS/c9/0M6tkt7fDbEQrwFADU8Tnr+BDdB/Hx+PMOJQbSFuzITxUH/uQr9tIHIwAH2NcTBh7iKBQQFvkf+CMOBt4N89bXDN0TDzD3BP4AN+EO4ScI6hvh/xYD9uVGMSxEMun9xR/5BuzM5c3wMbQqu+vMCAgcwgMaCgX9AjoT5wwt3CoaRRPv/f378ycV3A/oRBEJNQH7BfTS4TAgExktDyMHuQUIHd4GG/sH/u3c+ObtCu0IAO0XCPzm2b/e49ggBePtBef7EQbo6xYL9BPfEejmDukc7xf5NukZ1QD2Egf0COr/+gIU4RYfAOLOCCPs2REh3BbYDTHHGRD6KgEEBQ4eArzdCOz8WtD6Ifo4EBET+hjj5/cMB/zgyvnkAr8w6u4PAhzcFvrUChwODwnb0vsa/Ob+BNYq2/IN/hLw8ATa6O0sBfYy7hsuBw017gMG8CcjIcwKAyHmGur50PAV//QSGxrUDgXjBEkPEU3+FfX75BDTEh0O8O4m9vwZGRUECTQgGQod3vAmChexAQMjHdMx/Pbz/A0R4xn88glM+g4CBvv0QSzQFxMR9gjcy83cwwkkxxYc+fgLFsXVCDD48hcWwxRQ9yjw6jnNAOX3yfYQBiATCPn35hq0N0LcT/zP9Tji8RoB7BcdKwsx6BP30xgF1wL6Fg3+DPjLMQguBxJFBRf6/RES/hQGBtgbRA4KCzkX9cz2+wzqRP4O2+0VIQ8UGuA4Bx/YChonxTz+6hgM2QXzDuwEIdH61QwZ5jbpIPEAMg+hBkoM9v3SKcrz4e3xWhUIHQsY9NPl2tQO5Ofv6Mv9HwcC6eX+GfEZ18Ll8P7uOAsK+AXhAProu0PvFRH3Ev/tLg4ACAoM6hAY3e3vHyUvDtj3ASEI4s0OMvXb+usSCAIvzg0pJA4XCQvEHiy+1fv/mgMvCgQN4eD6DuXWDMzWEBEaNNtDAdXb8xYc3RIjCCM/6PIH+SkNPQgoCAEZ/gbyTQgZGfwQDAHJ+iLy+yXvAxc/8xoB2wH88wEm7f08MxD0FP0kBPj13+oN3NQKBev8+9H02PgXIQLq4hvS5NX61Oj28OX+MeYc/u/p8Ofm/AcSESshD/ElBfwTF+T77ADtQBYM6w3l//kcDdI2C/LcFgvn6N8i6hgDDQonNRLt4h73JgbjD9wt+uIKHOHpzeAABRfdACDzEvwFCZ8r3MkIGBjv4AO+8SS7HhHo2sIe9ODD/RgGL/T47sv3JBbxJyj4JvAI/dPc7O466xjSIxfd/RADIgcN2OjJ7wsJFRMt8uvbFSgEygI4AxYY0ucV9+XuIcU38BET/R3vEPHS7gjP+bLuQO3Z/d/6yf0R/AXk5egJxBX37BgT+N/r+QwULk/fPiAZwvP2BuwB+zj3DQUS2BNEIQMrK98cDggLDvsL9+r5A+UWtSrt/+DV/zTHIBn7CgQvHSoD7RvrEvQRygUd/DgA2frn7Pj119sy9QIiHPXj5htC2w/8IRX1GgDs8jrY3dwJ/SAE4RYZ/QQN/OooDt0X3P315AQHHxL92TMJDiDT7988D+UPNf76BejaGSHx5+EOIRACD/QsCN368f8FOB4AwggG6PEO7fwyDAr85vTxBxUM7+fKMSHS5gYU+wwMEjfqDMwrCgv+8NIUB9g6DgTe7eoRJ/fgBgc56BoA2ev8Gukx/9UPtgAD/ScF8AUMyTYp7i/l7t4A81P5H8rt3AUtJCL6KiVB9REYCzb95jDM9gf5yxsB3fDg8P0l3NHZ9OD8AflFEfvbIw4A2LTz+QIr8+sdC+b01vHoDAwC/PD+DgLf8Av26AXaM9HqCQYf3Qb3Afrt9gEeBCIE8dDd/xft+uUIET8HFvotI/dRFef8GgkV4ij8sevpQ+8lBOQx8OEJFgryLOQZBBcR8isf7AAa+/gSNecXCRr5AdAf7ccQ8+gHFATe8yTq9fQe2QryOerc2MILBSXQ/vADEtIEJ/XmzlT+APvUDw0u9v0fF/Qd8w7jH/QYLMUPDt8fFgoD2yAO6/r79uID5cDD4fPDzAjxEvgA1fL84QzC5tTwACICERYO6O77yib6HOwAHd7/+eoVIgAd8wXuJsIk1hT06tba++gE0xUAKd8n1+oCQg7p7e8BKQfI/QX5G8okADUK+xMGAyIWE/3UHtb6Dw0REAjY6fjtBPYGIgnp9CQ86+fsIc7C7vsfA9S06Af2/uEPKvka0gvm+DLu9xT1ERAevwTaIx796jQe4v3xwNkRIecZBwz4Aw3m2PMBFLn5LQ0QBgQhAQTYEsgVCgTwFwod2uAmCDQj3hDv/RoOLeMHGhUCDusECP0p6/ceCBAR9hPoA+EwEB/3seUrz/j/DOwGAvwB+Oke7QrsEfLoJg3LEBAr1uv+vRj8wgUeBADnCe3399UvMh33+OsA8Ln5J8QhQxYdAS/jrCHl1gcN7/gUSOzY2/TvEgcs7cY77SESFeEUzQv88in2GOsbMf77+NwC+vYl8SIUHfEe4Qb6/u4F+iED4isGDwv2QgAB+hL4OAzXAfk0OfIY7x4EAhv7FP/v+gw4Gw8t/hP33/cO+fP6AgrzFiMJ2dgwDNcK1SL+GBvrHvEGtgsGDOoaAvIV3AYf8/0xKBT0C/EQ//0mGTL++/Uv2L7eBBvp7gkzAeEe8SkBMvcaExD2LdwzQuEQ/vDs8gZTGggm1/vz7ST79SIXzgv7Cej32ewR4RcEAuvJzAwO/P7W89dJ7BXg7f7d7e30EtUa+A/z/BvdCw0SEOkgHfpILSEw6OgHKDLk3SIN9SAK+Q0Q5R8Cyekq3BnfGBnhHN4cLhUlCi8e8+8m//ca6iPNKtQn0QYp5O0FMT7x4RkX5PEBwPAO1yIoz+USzt3xDdESEiUN+iL/FS8W9fMRDREb/B8tHQAOJcUJrPAjKQj3Iv8P+ewB6RPM/BP22bwi7AoQ9R8B5ff/7uIN/gjrCfQRFyTm/iP8IvrQE9zY9+Af4OIH/vP19N1I7/ELD/QXRiYIBgsdAyjc3gYZL/BA5dbvBNsrzflGBifhxQL2/y4QT93P2AfdAuLx0+j/D/3hDx0NEBH+FhwCMwAKOQj9NyPnAhMM5QEQ9DIb9g7d9fnFEiPu6xTi5draFNsN9grq0hXmKgfhxjUv6+TV3hnm/9veOfvuBAUFBuXG4+oBJQHx7A0ZDPjR5h3b6e3kLLMh1xQT2hE2Rwzw+hUe+MwCBRIXCB7jJAzc8v7LDhjx+/gK1e3eDQ/rCBcOJrzN6fvo9BIK3hH91hstAPEs/+ohLefl7PgdDAroEQzN4wgYKwn0FAHxEv359/YXJ/kp/gHa+uzcJiHrEhzd4ywEBeYQ5eUU5QEOzjPzt+/+C9AKA+zA+RL3I9kc+iXaCf34ExfkTgEK/gnxH/8PMCUN/v8nCA0UEuMR7ewuFQwy/w8h/+bg+fjr+9bo//3wC8+84OYWD+7W9QoHMO3n2gL0Eh4N7u8RIdXoJTnAMPAhLM3f8DkFGAL/6gf7+P3vERHVJAnP+gEWC+rj7ObtLjnTLvrk9wnXJQX0XA34CRwhNPv8INfhA+fsDvvm6wEX0tLGugofwRa/+Rz/2tonLt4W7gw3FvEtIAmBDRIi3Oor9/re9Any//oVA+wDEvgG8hG59votCi7dxS0J6A7hER0QFMgGCvX4Ax7j3gz429vy2wcLACkL7hXjHSQH/+sv+eXuAS86BAwJCAE/L8AT1vTlBzQDGAgtBwEeyfDbFiIn+PMbDQoB1xcg7Sja0P0U4ukcGPPwFzYpHQ0GFy8a/vQtyB3t9Q3NBAfrEcoZVwn9GwrQF0Xq7Rj9HPTBBNv44zQEGNvl6/bG0vEm8Brp6iDzCObEBBf/2RLHHBbXICL0GxXv3Sfk5vbJTCEM+eT55NgW/CUY5fciIAvpDe4h7vr9xfUK5wPaOxkdKhT48DoX+SbeHUUR5x8SISIuGxUz8tZDDBYRMyr+MN3aBfUdFu4KCfwAziAP6gPlFCT+CSkc6ujx6fb+/wcwDfIH3vEA+Rj7EfIcCvQb8/QJyRkC8RkJFgYyJfTc4LwNAwDs9RkA5iwED+P43ljoEwj+HucTBhUSF+zf5gT58y0ZH+bu+88F6U3j/UKz0hgoEAwU5h/VONTq/xIE9/zvB9nnBgP29/oJ4f0EPPwgFxn6EyjrF/wiGfLXDBASB9rtBSvd4SP+CFYKEAj3F9Yx7PXpIQbrFAfxIvjo8/YwEh/f1MTW5+fo4M3fB+wG9ikE6N4E/AkcAfsT5hIG5PX56N0lPPDm5gviHuz/JSPsERH64BgX4fEc8y0EDO4k6x7OBwMbCPgMA9UNCeAnqV4F+SknEujV+wEf69j9DRMDyhP8xP4D9QkcGD0Q/iLq3e/R5eLw+vzpE+7cDuQJ7fLo6hX86NkrABDl08zjD9397gXuEuoH89cD5tr91e8c3uXSIBIlARMg2g8dzAra6uI67hcLIBrn1fM58MjeCwQ+IvbGBhDTCAAnGwnx6yLsDw0CDSsjBtQk9fnQ9igY9RH29wb3K++/8v0I5fHxCBI3Fw4y99bcIyoKAugVFwsC5CYLIwj87hrdIvz33xbr9P7lFZMmDvwK7BUJEBfvt9T+48TgIe/iz/Db5fjx8P3g6BgK7gH/AtPyDMYJGvcj5QAI1A0PQN71Du8KCtI4CvDkAu/I0eXVD+Pb2Rk+8+ETGPLzTfzt3vQc8xnu2Aa9CvH77A3v9g7P2x8tExIr8+UtEQfk+yEQHfj99d8Kwdkk8QUcIMgFCOcU/NXqHP0LEtcrBQErHM7jNP2/3+Mh+P/689EH6+vrHt8N+iz+I0EH+fwB9v3o+N0E4QXy6hH+2dwIMPQPAxf6HRfyBwTx9Qvs+zUFOxwiLkIhAeYX5+n44cTtACIvu+774AQi5P0B7Dz31epDEfBVWcQA2f/SEiX0DQX+9R/8DNToFOIyL+Dy4wTv6+QvN/L5PxX++DocFRMg2zXpDBYdEgQR8wQ1vfAM2d4JDSrn/yPYJNHpF/z8CAzEMQEvDU/SO/b5BPvh9c/p5QQeyQkMttv05fgOBu0oE/XW79gM8u8PANn/LwgCOBYo4Pay1QLlCxLC/csw4v8S5x3zKtwEAC0REwATIggv/vLNK90REMYkA/Ie1ujI/f3n2FHwBh4L0PMBvd/u0xMZG//M3/rwBxz5+PgQJA372egZCvfU7cb6EAgL7AMVBg3fDh8W8hTWA9jmCfztJfHXLwkU29o35NUX+QE0MfIjMtA3/QPh8v3uJQIANN8sMgvv8hvQMfHu5gQGCSEU4x0eGxzg8BrIEf8B6vkuGgfe6Pwa/wA3283xAxPcG/AD7RfSHiYM1DLnJAEP9vj/79wC7y0UKrf7BvDcCOMj1if39BUW6DASEQ4HEScaEwcw9+QkDQgd//sV6eQa9U4lBBTUGiwAGN30KRsPCdflCiYbJPnq4+rHCQYZGe/pEBr0BfrxHN367AntIxLl4PEA2Ab92fob9CofEdndFecE6s8L8wYOFTMQ9+TqxBjv/PH48RThLwsc0OEPC/YvCPsO/P8ULAUKLBIMBiDuEOD8JfHSGCf/8+w0/BXlE/gBDwQDEQwSBigmAxAUCu3O5u//JAA0E93tI9MDDu7wHvMJCslN2hnGFPjz+PvlHuznFiX38hj+BfPxByUN8hDRDe7g6Pjt/dMJ9u0hKQFEBrIvyQ0Y5SwBCB3b9ggK7C/mRfknIu0R5R4OCv8L6iHz5QrnxxAY+gDP9Agm5cwRNPEEwSbwBt42CwITAi8F7Act/t0J3wIbHejX6jAALQlL5vncGffa4uTsy+HaLwwC/d4dDR7w4NUo7gHfJBe/5fDQMeM6ENUVAQgBJQ3sBrwsANrrFQYf5RIJ4uIPFNM1/NzfGfUU/zcWKwMRJQEOCOz3ye//AQgP8CT0Ifz12+MrHxL1Hv/sABv1zuQV7LLn6CMTu+1N7OzuyAEJAffsEwMIGgTvFw8G7yr2uCzt+AYBDLP81Cvc/e0P7u7r5Ub72x03FRoRNwD76eIgBf71Ndq+7cva/CgSLgL9GQ7pBPgLDhnu0xgQ5/zny+oOGQdKNiQD5Bjx9gwTGAnc/sQFEQ8EIPbx8AnjL+je/v/SAg/mEPotAhnyKPX6+ssRJiPr5wM24wn19Rf26wPO/hDrDiMW1gDY3wf419jwLhw2wggk/OkI7AIFDebV0hQn2Rca8wcBvPbpIuok1/wv6/7zKyn/DPwE+wX33QTsAgD5EvTk3TL4BB0P4vk/9/vXLBf6GPEtLQfyC9H/AyPSCQMRBxrn2QcpBRPEGeDL3NHf1gr87icL3A0E/+//9AT5DjP0Duj9E+LP+xP49+cN7fQSEfI08/sZE/T8Fgjw9AzpADUs+gf28A47IBQM2y3h9yQSCsvfFN8H5hbx/fw5BPsGFSEC2wwj7zMLCEcEABjdEuIhFw7byuH9FPs/5/AQ"},"wpe":{"shape":[21,48],"scale":0.0234844,"b64":"0wcH3q/R6dBoScfpzfk1BPoyOgyjAPus3vMX/+weVRcJLfHI0BQMEOX25v8F+yMx6/2J6x7E1RUw7L3tAyMKa+Et+9cdH0AOQKJ2Qez/G67FDh708Crx10vGPL0szgTq1/U1OQgW2hDcx9rMHRMo/97pAwvG8+kLJUlK+AznGO/n6w8R6foqDfrs4gwRBAz/Qu0NKDDy2jsQPSsuzhguB6rvURsK8zfAIh/1tiPYKLD8IRXFFRAYNjflFdIuAcXe//ERDNGnDhES8R8G+/w7GyEd7xq30v4nZNAs9QDoUAjiwu8q2i8A7wnsDwAP7Pvc2EkC+vHWBwXTvBjrygAm774iu7IS7y3BDvwiBeEQ1+T9F+A3C/IoGzgqIPsZ5O3n+dvuLgkAIPEo46ghD+IG3TYJMbb13+7a/fIdON0J/+3V3LUeyS8dBv0JGyYDKdG/NvoZzcauvDn8BND20Kso79kj+ac65/vau/jv7uji/BLHJCXE7wKUv7068Bn5E0ncBTMfB/jnARHqAvTf/rLCzQkLBL1ZTf/vCgHz6QrOLDVcFt7VDvLjExj/SUTpCRoS3/TbAij3p9vi+Mr6LPP4TuzgDPf0EfcAkx4B38fjEA/k9kn87UwRDB8mx8kN887hGCEGGjTsAxTq9gP4xyPFAR/f7tP/NfX5Ft+jruX368U0+e4w/9cQKh4QBgTb5x0j9ugARKHcL6wE8f7kR+j/9hARAuSuKwOr5vP68f/6LgUdxT4tBjUnu9voIAZHBvfqGt8JF/sRSyv++9AZ4A/LJR74+wQDTBDQAiEFC+8jLB8fDA0u9xnzyiWs5PbGOYu/3v4XDur67AXpzf0bMzLkBA7r/EwL1Ord8OvgERcGA7XPEBPK0u/lOwjWRQdJCj+78ATdYvIT+/XrvcIWsOvIRd0WEOvn7uvGCivQRwsoyzjt9d+cFiDrBDHsJ8wB6yEwKtXP2w829tED9vQv6xsm6E38H8iqN1neAgHlBdYDHuwb8w4B9Dco3D3VHRg8KywMDrYKOcor/mYJ+OP/yPg37jMGUL79QM48CujwRega0lgAOkwRJxgb2PQhJTpY2g3Jw+rsSgkZEvX9FRoZIecx9SX8QykW4hvWSCjjAwC5SBUvEAAyA+wL9fwG0gisIwU3/hwtUBHZ0QcREzjW/sEWIyYZ9UKtIh4euwITGiPs/UAIJ6g901LOKCPJVBPJR9UIuP39BvQI7xbdJdZG5hBw/9cp9BL6SunT7WD5Vb/tEx0i4uT9Ww6dOs3jMijsJ/g2BOUTE+JxMhsHLebyACL4NOlLPxItzX+p6doGLjsr484vAOgJpwwVzvovGyUxGToQ"},"ln1":{"shape":[48],"scale":0.0092174,"b64":"d3Z1dHBwentyd2t/e2l6emtlfXJyf31+eHdte251fX5/cXt4enR3d3d5entxe2t7"},"wq":{"shape":[48,48],"scale":0.0035174,"b64":"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"},"wk":{"shape":[48,48],"scale":0.0039291,"b64":"xNzX8Pnfyv8uHQ/JGuoMGOX+KeI6S+IMIUfmHNLhM9vg4A0X7C0tGOIc3iK3NznvHOT79/1GNtL29/gk1N8O7u/FACr9Te7QwB4F99MT9Csb0xoNEuHrGgQT1ABTAhbP2ptFHdkN9Dmqw+kMF58AxyzZ9wrpxPXU0wO/9R0h+kcxDAMn7cnvCuIG/CdGMtRF6R70CAUI4wnq2EOqsREE4UAB6sT2IcI2PObbFHI1RfgeJsRB1CNELBrLChbDyNtQFcQ22/7qA/gKEudYFDd07urVANj4Gy3x8lxF0Qfq8tgCCSwB3z4VOfYl/U4bQSZF/RIB0bj2NgkfE8Gr9CISTvgaxcb+Qx+sHxb19gwGrRfnvwXy6x0NNvEZ5zxIKSswQwgQ1RlEJhwA/L72DRjo5fbQIxxPL8PcxtcIZa34XEntNwjB/vz34yyT9iMmS0raC90d+wwxA8Yc+kpcA8UW1vP/0Arhv+vc1fMeUBkz8tsAKjcf/R/8xP8uBt/j+guOJ/X6Cggf+8EY4ShZuiQAGDoUGxL8zxDq2Pc0wvtO5ALC1RPz+S5B4ALaNrXiJfD4z8T1wdroFNlGLSm5I8EMIDcWyMsDQhbx3TPk8d64MRkJ8DUmwToM+9DvMhETAyLf5f8k2Mzb+9HLTTQIFwX0KAe2q+w6TC+29gm578bwJzsgDiH+9h3cTiPi3Bk03yga47GyPwEM/tjj7uLDDAjs7Vq8JAn8kRQ46vxCKhX1GiLaqQ3300D89Af09xXb7AAL4O4aHkz/CD8eE/RJCqAdJALbStk29dsoAvG5TNohpgo9Eu3/ECL+rixcP+PH+B/R7O8VDfkEqyrxGeYM/SDb7xEfICLfHRwhIiLOBzHGMuc0GxsyFwPQ//3P/d5g8Nz5CuvB2ShA7hTm8jcBEPzpBtkF6d01GAjqDfk5NDfuteQa6UgD7vX4LgXQGNM3Bfjf+gM58AAa/vLN9uwH4fgS3cY/PushLS/uDRkSH/sa0wAoCMUHjOEyNB0G9cMYGRDMJQUu3iAf2+vHANYr3twGt+fk5O0WCQ4+yCvtVdMRFCQcEh6+5w652zz28xwkCgym+ufy7iHwDw/qAAmuUdzASP3IAhHs+eH76ujhEirbHuha4iYhUbAQ+vwq7csPATsEON0J9vO4LAjb9kXfJAnI3O32CSQFNwr+4CfKwukCD81OCB8TGTG2DfLzHd0J4zAe/b/o+f3xMuQaKAVXCwmW0DkZ2Oop2/UTz8ThDdvvMQX98+4AM/a+LOfkzSMNGiG4AhPdN/gK3CQCHh0EJSS03PvlIiYp/PkgF/LVrh3q0605FQMtHtH65jwKLxAKKR9BT+MPLQb6Mzz2HcAwERvSIfb+LMIHCjDx8MHdFzz73s8+/OYz8/TmBjnu7OUXHPEHtj7Z6ebK8urt5Gfu/gkmSND7+Lwc+N6+NNccQv051FbR8cL+EkrYEwAO8eYDojL53Sjn9s4o4+3jKEaeAjk08skcJwvNskk8TdoKuCzmMTjrBefmpTYi+vQxJ9JIw/P6/enzOx0uGuQfFNzd/tDlygXspRITOiRLwwUz3+G1sg37FmUnERn82rA4Kg4fsgPx7CcxAs7L6hYW4Vzhv/3lPO4GGP4bAQ+7/fccG/Xn/fEHQRsOYLwNw0Od1/npGz/t/fjt3DIHpe/cHQUOHP0aE8w0+DMKUucODyPZwgceAPoeGBTuLRMExQAz8tmqDx8ZCA/WytGbDKoFKtQI8APK9+2+AOb429jUDb7jJgT7/DFLO9evFuoRBvT8b/r3Ter+zf1O8wY0RSIAIFD3rubdwrjhyxQkwAj6UgIJ3yA5HQnvHCX43DgExCLmu94G+jSzALC6CtrZBwsUE87ZFewnG+4VEAQRNUAW6T3+Reky3dI9pPEYaCjyU+zEEgpSHPrtOUwquEjMEDPjPdMgGxBGTK8w6PMWEBk+Kj0jxRT4I9VX5u9WCfHg+QcRNwQRJ9jL3hsQxPwnQPj5S8XjzeYd1QEH8v/UYPSvu/PrA8jIDRm1CLwP2eI04wkmFRgwFygYyS765P//3fgLNMndubr8tx/uCxvRB7n91REo4fPXJ/fpBhTG7g0J6/AOr7YB4AEWOyHi/Q5NJBoFJufe2upET+oB3QIBERf9CgxaCfHB5xYDIzAS4w3DBeSuNTX87tPVz9ME4Rj90AfmLuMbRwjO/DMsJ7scGgLNWAXbCeoTUcfnOBKsx1YTuQKz47ozMqjyuDbZB/85ngEhCOsYCZRHDhBZ5Bz8Zf0GRjwF9eHEHuNK7NMfPq8LFhIh7//a1cbGCt/PKgwGR8b/TKn1wbU01eTruNC2F/7kDSLeMNQC8c6uBOIw69EFf6oUUNU4v9P+KSjzFRzyQacfWRYmSirpCCbHGq0A5iNZDBclMRM52OLpPiG6OyStSt05D+PSxRb98izwSevaN4fqKMHhLbLy/C1B3wwUudjvEBEIKvQC9OBAEuc94xhBMgrlL+3gK+cN/Rq/qdxQqqUxLhkvNjVDSrkE41/H+Mhf+xZVD+z19eDyCxcG8PIJ7ine2K40tig3Ke8ltRHpYBkIMPLWIQvCIj3M787myCA68u0e/EYyys/W/cvTJdjFkt9OV0An3jPm/uvnYigD/gHM4qoP5lXQxe4z3xVHKNh8JA9KHfwH0NkQCNPpOvocANxhUOswIggSLeQmHLX4KAMeFEIb/SUqxTa3KwPApOg3JVEBBzw1IwlB/wrXJl0JKgQI4zwUnAJFFDng2dsBv+OryvZOtQ7b1OYLoAe+AP/wzAP7zsGsMvrbQN0l2vL3OvIX27o56iFgJOXQMfkINxykuw4P6xreEP/0CxMZK9P1LiXz+Czo5vgE+yYXvi8iBPin3gk0/j4lztvpTjgJ7f3ZAgsY7As2/SZARRncNCjVPQ8YC7b7v+vsAr/ry8QSH/uaMwZR7P02IzjgcQH7sr4z2UEdsuvbpiML+CEh3vc3IkYMGcYg4tckDkDG7QfyDPPk7dnhvCH74NH6BGPHJTDKQwz90EfRAOrNKg0Y5zbIRi7/sxMIGcC7+cje5NnIDdkJMjojKwbd"},"wv":{"shape":[48,48],"scale":0.003689,"b64":"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"},"wo":{"shape":[48,48],"scale":0.0036087,"b64":"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"},"ln2":{"shape":[48],"scale":0.0097717,"b64":"emx2e3dzeGtlfnlzc2h7f21sdnVwZndtdGxlcHJzc2l0dnRzbnV/dXF0eHp/fGhs"},"fc1_w":{"shape":[96,48],"scale":0.0039262,"b64":"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"},"fc1_b":{"shape":[96],"scale":0.0022424,"b64":"ABHastTV9Qwv9OIa+v7OGd/C7yW/Bcqw3b7OAw1AFNyZR9Xx1TGp4BSgwiEHDqEzvis5x9m0yMwrC/ylRNIZJgsg6xGRv0XtBZv0TfWB/vH5HMM8GtH6/eUBQK3j7vYe"},"fc2_w":{"shape":[48,96],"scale":0.0038019,"b64":"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"},"fc2_b":{"shape":[48],"scale":0.0017307,"b64":"dmA4vdol8iQ6v38R/vyXx21tQCXKJ/Ax4QRCUzUrAVdtWO3zHKrVbc3yS7xUM8UV"},"ln_f":{"shape":[48],"scale":0.0108856,"b64":"dn97eHp2eXp5enl/eHp5en58eXh4fHR3enp+enp9enp7enh7fHx3fHd4e3x6fHp6"},"lm_head":{"shape":[307,48],"scale":0.0051171,"b64":"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"}}};

// Decode Base64 Int8 tensors into typed Int8Arrays on startup
function unpackInt8Tensor(spec) {
  const bin = atob(spec.b64);
  const q = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const code = bin.charCodeAt(i);
    q[i] = code >= 128 ? code - 256 : code;
  }
  return { q, scale: spec.scale, shape: spec.shape };
}

const ENC_WEIGHTS = {
  fc1_w: unpackInt8Tensor(MODEL_BUNDLE.encoder.fc1_w),
  fc1_b: unpackInt8Tensor(MODEL_BUNDLE.encoder.fc1_b),
  fc2_w: unpackInt8Tensor(MODEL_BUNDLE.encoder.fc2_w),
  fc2_b: unpackInt8Tensor(MODEL_BUNDLE.encoder.fc2_b),
};

const DEC_WEIGHTS = {
  wte: unpackInt8Tensor(MODEL_BUNDLE.decoder.wte),
  wpe: unpackInt8Tensor(MODEL_BUNDLE.decoder.wpe),
  ln1: unpackInt8Tensor(MODEL_BUNDLE.decoder.ln1),
  wq: unpackInt8Tensor(MODEL_BUNDLE.decoder.wq),
  wk: unpackInt8Tensor(MODEL_BUNDLE.decoder.wk),
  wv: unpackInt8Tensor(MODEL_BUNDLE.decoder.wv),
  wo: unpackInt8Tensor(MODEL_BUNDLE.decoder.wo),
  ln2: unpackInt8Tensor(MODEL_BUNDLE.decoder.ln2),
  fc1_w: unpackInt8Tensor(MODEL_BUNDLE.decoder.fc1_w),
  fc1_b: unpackInt8Tensor(MODEL_BUNDLE.decoder.fc1_b),
  fc2_w: unpackInt8Tensor(MODEL_BUNDLE.decoder.fc2_w),
  fc2_b: unpackInt8Tensor(MODEL_BUNDLE.decoder.fc2_b),
  ln_f: unpackInt8Tensor(MODEL_BUNDLE.decoder.ln_f),
  lm_head: unpackInt8Tensor(MODEL_BUNDLE.decoder.lm_head),
};

const ENC_TOK_MAP = new Map();
MODEL_BUNDLE.enc_vocab.forEach((w, i) => ENC_TOK_MAP.set(w, i));

const DEC_TOK_MAP = new Map();
MODEL_BUNDLE.dec_vocab.forEach((w, i) => DEC_TOK_MAP.set(w, i));

const RECENT_VARS = {};
for (const [intent, count] of Object.entries(MODEL_BUNDLE.intent_num_vars)) {
  RECENT_VARS[intent] = new Array(count).fill(false);
}

/* =========================================================
 * INT8 MATRIX-VECTOR & TRANSFORMER KERNELS (W8A32)
 * ========================================================= */
function matVecInt8(x, W, b = null) {
  const [outDim, inDim] = W.shape;
  const out = new Float32Array(outDim);
  const q = W.q;
  const wScale = W.scale;

  for (let o = 0; o < outDim; o++) {
    let acc = 0;
    const rowOff = o * inDim;
    for (let i = 0; i < inDim; i++) {
      acc += x[i] * q[rowOff + i];
    }
    let val = acc * wScale;
    if (b) val += b.q[o] * b.scale;
    out[o] = val;
  }
  return out;
}

function siluInPlace(x) {
  for (let i = 0; i < x.length; i++) {
    x[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return x;
}

function rmsNorm(x, lnW) {
  const d = x.length;
  let ss = 0;
  for (let i = 0; i < d; i++) ss += x[i] * x[i];
  const invRms = 1 / Math.sqrt(ss / d + 1e-5);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    out[i] = x[i] * invRms * (lnW.q[i] * lnW.scale);
  }
  return out;
}

/* =========================================================
 * STAGE 1: PYTORCH INT8 NEURAL PROMPT ENCODER
 * ========================================================= */
const STOPWORDS = new Set([
  "is", "are", "the", "a", "an", "to", "for", "of", "in", "on", "at",
  "does", "do", "did", "has", "have", "can", "could", "would", "should",
  "i", "me", "my", "he", "him", "his", "it", "this", "that", "or", "and",
  "what", "whats", "who", "where", "how", "why", "when", "with", "from", "about"
]);

function editDistLe1(a, b) {
  if (Math.abs(a.length - b.length) > 1) return false;
  if (a === b) return true;
  if (a.length === b.length) {
    let diffs = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i] && ++diffs > 1) return false;
    }
    return true;
  }
  const [longer, shorter] = a.length > b.length ? [a, b] : [b, a];
  let i = 0, j = 0, diffs = 0;
  while (i < longer.length && j < shorter.length) {
    if (longer[i] !== shorter[j]) {
      if (++diffs > 1) return false;
      i++;
    } else {
      i++;
      j++;
    }
  }
  return true;
}

function inferPromptIntent(text) {
  const t0 = performance.now();
  const toks = text
    .toLowerCase()
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const bow = new Float32Array(MODEL_BUNDLE.enc_vocab.length);
  let matchedContent = 0;
  let matchedBigrams = 0;

  const negSet = new Set(["dont", "not", "never", "no", "nah", "nope", "cancel", "stop", "hate"]);
  const hasNegation = toks.some((t) => negSet.has(t));

  // Direct intent check for fun / game requests (when not negated)
  const gameKeywords = new Set(["fun", "game", "games", "bored", "connect", "entertain"]);
  if (!hasNegation && toks.some((t) => gameKeywords.has(t))) {
    return { intent: "play_game", prob: "0.99", dtMs: (performance.now() - t0).toFixed(2) };
  }

  // Section-specific intent check: experience vs projects vs both
  const hasExp = toks.some(
    (t) =>
      t.startsWith("exper") ||
      t === "jobs" ||
      t === "career" ||
      t.startsWith("intern") ||
      t === "worked" ||
      (t === "work" && !toks.includes("open") && !toks.includes("outside") && !toks.includes("how"))
  );
  const specificProjWords = new Set(["voice", "chorus", "boson", "openclaw", "pytorch", "encoder", "speech", "youtube", "ml", "arch", "architecture", "tinygpt"]);
  const hasSpecificProj = toks.some((t) => specificProjWords.has(t));
  const hasProj = toks.some((t) => t.startsWith("proj")) && !hasSpecificProj;

  if (hasExp && hasProj && !hasNegation) {
    return { intent: "explain_all", prob: "0.99", dtMs: (performance.now() - t0).toFixed(2) };
  }
  if (hasExp && !hasProj && !hasNegation) {
    return { intent: "explain_experience", prob: "0.99", dtMs: (performance.now() - t0).toFixed(2) };
  }
  if (hasProj && !hasExp && !hasNegation) {
    return { intent: "explain_projects", prob: "0.99", dtMs: (performance.now() - t0).toFixed(2) };
  }

  for (const t of toks) {
    if (ENC_TOK_MAP.has(t)) {
      const w = negSet.has(t) ? 3.5 : 1.0;
      bow[ENC_TOK_MAP.get(t)] += w;
      matchedContent++;
    } else if (t.length >= 4 && !STOPWORDS.has(t)) {
      for (let i = 0; i < MODEL_BUNDLE.enc_vocab.length; i++) {
        const k = MODEL_BUNDLE.enc_vocab[i];
        if (!k.includes("_") && k.length >= 4 && (editDistLe1(t, k) || t.startsWith(k) || k.startsWith(t))) {
          bow[i] += 0.85;
          matchedContent++;
          break;
        }
      }
    }
  }

  for (let i = 0; i < toks.length - 1; i++) {
    const bg = toks[i] + "_" + toks[i + 1];
    if (ENC_TOK_MAP.has(bg)) {
      const w = bg === "do_not" || bg === "dont_show" || bg === "dont_explain" ? 3.5 : 1.5;
      bow[ENC_TOK_MAP.get(bg)] += w;
      matchedBigrams++;
    }
  }

  if (hasNegation && ENC_TOK_MAP.has("dont")) {
    bow[ENC_TOK_MAP.get("dont")] += 3.5;
    matchedContent++;
  }

  // OOD Guard: if there are unknown non-stopword tokens and zero matched content tokens, reject
  const unknownContent = toks.filter((t) => !STOPWORDS.has(t) && t !== "you" && t !== "your");
  if (matchedContent === 0 && (matchedBigrams === 0 || unknownContent.length > 0)) {
    return { intent: "unmatched", prob: "0.00", dtMs: (performance.now() - t0).toFixed(2) };
  }

  let norm = 0;
  for (let i = 0; i < bow.length; i++) norm += bow[i] * bow[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < bow.length; i++) bow[i] /= norm;
  }

  // Forward pass: Linear1 -> SiLU -> Linear2 -> Softmax
  const h = siluInPlace(matVecInt8(bow, ENC_WEIGHTS.fc1_w, ENC_WEIGHTS.fc1_b));
  const logits = matVecInt8(h, ENC_WEIGHTS.fc2_w, ENC_WEIGHTS.fc2_b);

  let maxL = -Infinity;
  let bestIdx = 0;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxL) {
      maxL = logits[i];
      bestIdx = i;
    }
  }

  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) sumExp += Math.exp(logits[i] - maxL);
  const prob = 1 / sumExp;

  const dtMs = (performance.now() - t0).toFixed(2);
  return {
    intent: prob >= 0.25 ? MODEL_BUNDLE.intents[bestIdx] : "unmatched",
    prob: prob.toFixed(2),
    dtMs,
  };
}

/* =========================================================
 * STAGE 2: CAUSAL TRANSFORMER FORWARD PASS WITH KV-CACHE
 * ========================================================= */
function sampleVariationIndex(intent) {
  const numVars = MODEL_BUNDLE.intent_num_vars[intent] || 1;
  if (numVars <= 1) return 0;

  const mem = RECENT_VARS[intent];
  if (mem.every(Boolean)) mem.fill(false);

  const candidates = [];
  for (let i = 0; i < numVars; i++) {
    if (!mem[i]) candidates.push(i);
  }
  const chosen = candidates[Math.floor(Math.random() * candidates.length)];
  mem[chosen] = true;
  return chosen;
}

function transformerStepInt8(tokId, pos, kCache, vCache) {
  const D = MODEL_BUNDLE.meta.d_model;
  const H = MODEL_BUNDLE.meta.n_heads;
  const Dh = D / H;

  // 1. Token + Positional Embedding Lookup (Dequantized from Int8)
  const x = new Float32Array(D);
  const wteOff = tokId * D;
  const wpeOff = pos * D;
  for (let d = 0; d < D; d++) {
    x[d] =
      DEC_WEIGHTS.wte.q[wteOff + d] * DEC_WEIGHTS.wte.scale +
      DEC_WEIGHTS.wpe.q[wpeOff + d] * DEC_WEIGHTS.wpe.scale;
  }

  // 2. Pre-RMSNorm + Q, K, V Projections
  const h1 = rmsNorm(x, DEC_WEIGHTS.ln1);
  const q = matVecInt8(h1, DEC_WEIGHTS.wq);
  const k = matVecInt8(h1, DEC_WEIGHTS.wk);
  const v = matVecInt8(h1, DEC_WEIGHTS.wv);

  kCache.push(k);
  vCache.push(v);

  // 3. Causal Multi-Head Self-Attention over [0 .. pos]
  const T = kCache.length;
  const y = new Float32Array(D);
  const scale = 1 / Math.sqrt(Dh);

  for (let h = 0; h < H; h++) {
    const hOff = h * Dh;
    const scores = new Float32Array(T);
    let maxS = -Infinity;

    for (let t = 0; t < T; t++) {
      let dot = 0;
      const kt = kCache[t];
      for (let d = 0; d < Dh; d++) {
        dot += q[hOff + d] * kt[hOff + d];
      }
      const s = dot * scale;
      scores[t] = s;
      if (s > maxS) maxS = s;
    }

    let sumExp = 0;
    for (let t = 0; t < T; t++) {
      scores[t] = Math.exp(scores[t] - maxS);
      sumExp += scores[t];
    }

    for (let t = 0; t < T; t++) {
      const att = scores[t] / sumExp;
      const vt = vCache[t];
      for (let d = 0; d < Dh; d++) {
        y[hOff + d] += att * vt[hOff + d];
      }
    }
  }

  // Output projection + Residual connection
  const attnOut = matVecInt8(y, DEC_WEIGHTS.wo);
  for (let d = 0; d < D; d++) x[d] += attnOut[d];

  // 4. Pre-RMSNorm + SwiGLU/SiLU MLP + Residual connection
  const h2 = rmsNorm(x, DEC_WEIGHTS.ln2);
  const mlpHid = siluInPlace(matVecInt8(h2, DEC_WEIGHTS.fc1_w, DEC_WEIGHTS.fc1_b));
  const mlpOut = matVecInt8(mlpHid, DEC_WEIGHTS.fc2_w, DEC_WEIGHTS.fc2_b);
  for (let d = 0; d < D; d++) x[d] += mlpOut[d];

  // 5. Final RMSNorm + LM Head (x_final @ lm_head^T)
  const xFinal = rmsNorm(x, DEC_WEIGHTS.ln_f);
  return matVecInt8(xFinal, DEC_WEIGHTS.lm_head);
}

function generateWithMicroGPT(intent) {
  const t0 = performance.now();
  const varIdx = sampleVariationIndex(intent);
  const seqTok = `<seq:${intent}:${varIdx}>`;
  if (!DEC_TOK_MAP.has(seqTok)) {
    return {
      words: [
        "Not", "sure", "about", "that—here's", "what", "I", "can", "do:",
      ],
      genMs: "0.10",
    };
  }

  const kCache = [];
  const vCache = [];
  const eosId = DEC_TOK_MAP.get("<eos>");
  const maxLen = MODEL_BUNDLE.meta.max_len;

  // Prefill <seq:intent:var> prefix token into the Transformer KV-cache
  let logits = transformerStepInt8(DEC_TOK_MAP.get(seqTok), 0, kCache, vCache);

  const generatedWords = [];
  let pos = 1;

  while (pos < maxLen) {
    let bestId = 0;
    let bestVal = -Infinity;
    for (let v = 0; v < logits.length; v++) {
      if (logits[v] > bestVal) {
        bestVal = logits[v];
        bestId = v;
      }
    }

    if (bestId === eosId) break;
    generatedWords.push(MODEL_BUNDLE.dec_vocab[bestId]);

    logits = transformerStepInt8(bestId, pos, kCache, vCache);
    pos++;
  }

  const genMs = (performance.now() - t0).toFixed(2);
  return { words: generatedWords, genMs };
}

/* =========================================================
 * DOM MUTATORS & PAGE STATE
 * ========================================================= */
const EXP_IDS = ["exp-uoft", "exp-mpbsdp", "exp-mgcs"];
const PROJ_IDS = ["proj-chorus", "proj-arch", "proj-tinygpt", "proj-transformer", "proj-speech"];
const ALL_IDS = [...EXP_IDS, ...PROJ_IDS];
const expandedRows = new Set();
let currentRunId = 0;
let lastQueryCmd = "";

const SIMPLE_COPY = Object.fromEntries(
  ALL_IDS.map((id) => [
    id,
    document.querySelector(`#${id} .row-desc`)?.textContent.trim() || "",
  ])
);

const COPY_STATE = {
  extended: {
    "exp-uoft": "Built an agentic RAG service including a RAG pipeline, hybrid graph/vector retrieval with temporal information based on Graphiti, using Neo4j, pgvector and postgresql. Also built multi-agent search features using Langchain and Claude SDK as well as MCP and API endpoints for a source-of-truth database of product information.",
    "exp-mpbsdp": "Modernized Ontario's corporate payment platform (Transfer Payment Ontario) with Java, Spring Boot, Angular, and TypeScript, designing REST APIs for enterprise payment workflows. Also automated applicant funding-viability scoring for Experience Ontario using Oracle SQL and led a team of interns building an internal program-assessment web app.",
    "exp-mgcs": "Built accessible, AODA/WCAG-compliant web frontends in React and AngularJS serving 100+ ministry clients across Ontario. Also developed internal Java and XML administrative tools in Eclipse and ran regression testing across agile sprints.",
    "proj-chorus": "Built for the Boson AI Hackathon — I designed and implemented the multi-agent overseer architecture for a real-time voice harness powered by Boson Higgs Realtime 1.5. An overseerlayer running on Higgs 1.5 realtime (which has async tool calling) monitors the conversation and live routes conversation context to audio based agents, which leads to seamless multi-agent voice interactions and context handoffs.",
    "proj-arch": "Replicated the ArchAgent paper (arXiv 2601.13007) to recover high-level software architecture from monorepo-scale codebases. Uses tree-sitter for static AST dependency extraction, adaptive file clustering, and multi-agent context compression so large repositories fit cleanly inside LLM context windows.",
    "proj-tinygpt": "I trained a two-stage 67K-parameter language model in PyTorch on a synthetic corpus of common phrases, combining a BoW/bigram neural intent encoder with a 4-head causal Transformer decoder. Exported via symmetric Int8 weight quantization to run sub-millisecond autoregressive token generation, DOM tool calls, and a minimax Connect Four engine in pure frontend JavaScript.",
    "proj-transformer": "I wanted to understand how LLMs work so I built and pre-trained a single-head self-attention encoder-only Transformer from scratch in PyTorch and PyTorch Lightning. Implemented a custom masked language modeling (MLM) data collator and training pipeline to learn contextual token representations.",
    "proj-speech": "I was curious on how LLMs affect everyday language, so I built a Python data pipeline using Pandas, NumPy, and SciPy and analyzed 10,000+ minutes of YouTube speech across 20+ content categories to measure how LLM-favored vocabulary is entering everyday human speech. Quantified word-frequency shifts using smoothed log-odds ratios with Woolf's standard error, Bonferroni-corrected two-proportion z-tests, and paired t-tests."
  }
};

function mutateRowsDetail(ids, mode) {
  const map = mode === "simple" ? SIMPLE_COPY : COPY_STATE[mode];
  for (const id of ids) {
    const el = document.querySelector(`#${id} .row-desc`);
    if (el && map[id]) el.textContent = map[id];
    if (mode === "extended") expandedRows.add(id);
    else expandedRows.delete(id);
  }
}

function collapseAllRows() {
  mutateRowsDetail(ALL_IDS, "simple");
  clearHighlights();
}

function mutateTheme(preset) {
  const root = document.documentElement;
  if (preset === "dark") {
    root.style.setProperty("--bg", "#141414");
    root.style.setProperty("--text", "#ededed");
    root.style.setProperty("--muted", "#8e8e8e");
    root.style.setProperty("--border", "#2a2a2a");
  } else if (preset === "light") {
    root.style.setProperty("--bg", "#ffffff");
    root.style.setProperty("--text", "#37352f");
    root.style.setProperty("--muted", "#787774");
    root.style.setProperty("--border", "#ededeb");
  } else if (preset === "paper") {
    root.style.setProperty("--bg", "#f7f3eb");
    root.style.setProperty("--text", "#2c2623");
    root.style.setProperty("--muted", "#7a7067");
    root.style.setProperty("--border", "#e6dfd3");
  } else if (preset === "terminal") {
    root.style.setProperty("--bg", "#0a0f0d");
    root.style.setProperty("--text", "#4af626");
    root.style.setProperty("--muted", "#2ea818");
    root.style.setProperty("--border", "#193814");
  }
}

function clearHighlights() {
  document.querySelectorAll(".row").forEach((el) => {
    el.classList.remove("dom-highlight");
    el.style.opacity = "1";
  });
}

function focusRows(ids) {
  document.querySelectorAll(".row").forEach((el) => {
    if (ids.includes(el.id)) {
      el.classList.add("dom-highlight");
      el.style.opacity = "1";
    } else {
      el.classList.remove("dom-highlight");
      el.style.opacity = "0.4";
    }
  });
}

function clearExtras() {
  const helpEl = document.getElementById("agent-help-list");
  if (helpEl) helpEl.remove();
  const c4El = document.getElementById("c4-widget");
  if (c4El) c4El.remove();
}

function clearAgent() {
  currentRunId++;
  lastQueryCmd = "";
  document.getElementById("agent-input").value = "";
  document.getElementById("agent-trace-container").classList.add("hidden");
  document.getElementById("agent-clear").classList.add("hidden");
  clearExtras();
  collapseAllRows();
}

function triggerCmd(cmd) {
  const input = document.getElementById("agent-input");
  input.value = cmd;
  if (cmd === "tinygpt") {
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (typeof input.focus === "function") input.focus();
  }
  runAgentQuery(null);
}

function renderHelpList(container) {
  const list = document.createElement("div");
  list.id = "agent-help-list";
  list.className = "agent-help-list";

  const items = [
    { cmd: "explain experience", desc: "expand all work experience descriptions" },
    { cmd: "explain projects", desc: "expand all project descriptions" },
    { cmd: "keep it simple", desc: "restore short 1-sentence summaries" },
    { cmd: "play connect four", desc: "play Connect Four against the model" },
    { cmd: "what should i cook", desc: "sample a favorite recipe recommendation" },
    { cmd: "focus on agents", desc: "highlight AI agent & RAG work" },
    { cmd: "dark mode", desc: "switch theme (or 'light mode', 'paper theme')" },
    { cmd: "copy email", desc: "copy jeffersonn.liu@gmail.com to clipboard" },
    { cmd: "reset", desc: "restore page to default" },
  ];

  for (const item of items) {
    const row = document.createElement("div");
    row.innerHTML = `· <span class="agent-cmd" onclick="triggerCmd('${item.cmd}')">${item.cmd}</span> — ${item.desc}`;
    list.appendChild(row);
  }
  container.appendChild(list);
}

/* =========================================================
 * CONNECT FOUR ENGINE & CENTERED HIGH-CONTRAST BOARD
 * ========================================================= */
const C4_ROWS = 6;
const C4_COLS = 7;

function c4GetLowestRow(grid, col) {
  for (let r = C4_ROWS - 1; r >= 0; r--) {
    if (grid[r][col] === 0) return r;
  }
  return -1;
}

function c4CheckWin(grid, player) {
  // Horizontal
  for (let r = 0; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      if (
        grid[r][c] === player &&
        grid[r][c + 1] === player &&
        grid[r][c + 2] === player &&
        grid[r][c + 3] === player
      ) {
        return [[r, c], [r, c + 1], [r, c + 2], [r, c + 3]];
      }
    }
  }
  // Vertical
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c < C4_COLS; c++) {
      if (
        grid[r][c] === player &&
        grid[r + 1][c] === player &&
        grid[r + 2][c] === player &&
        grid[r + 3][c] === player
      ) {
        return [[r, c], [r + 1, c], [r + 2, c], [r + 3, c]];
      }
    }
  }
  // Diagonal down-right
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      if (
        grid[r][c] === player &&
        grid[r + 1][c + 1] === player &&
        grid[r + 2][c + 2] === player &&
        grid[r + 3][c + 3] === player
      ) {
        return [[r, c], [r + 1, c + 1], [r + 2, c + 2], [r + 3, c + 3]];
      }
    }
  }
  // Diagonal up-right
  for (let r = 3; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      if (
        grid[r][c] === player &&
        grid[r - 1][c + 1] === player &&
        grid[r - 2][c + 2] === player &&
        grid[r - 3][c + 3] === player
      ) {
        return [[r, c], [r - 1, c + 1], [r - 2, c + 2], [r - 3, c + 3]];
      }
    }
  }
  return null;
}

function c4ScoreWindow(w, player) {
  const opp = player === 2 ? 1 : 2;
  let pCount = 0;
  let oCount = 0;
  let empty = 0;
  for (let i = 0; i < 4; i++) {
    if (w[i] === player) pCount++;
    else if (w[i] === opp) oCount++;
    else empty++;
  }
  if (pCount === 4) return 1000;
  if (oCount === 4) return -1000;
  if (pCount === 3 && empty === 1) return 14;
  if (pCount === 2 && empty === 2) return 4;
  if (oCount === 3 && empty === 1) return -18;
  if (oCount === 2 && empty === 2) return -3;
  return 0;
}

function c4EvaluateBoard(grid) {
  let score = 0;
  // Center column preference
  for (let r = 0; r < C4_ROWS; r++) {
    if (grid[r][3] === 2) score += 4;
    else if (grid[r][3] === 1) score -= 3;
  }
  // Windows
  for (let r = 0; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      score += c4ScoreWindow([grid[r][c], grid[r][c + 1], grid[r][c + 2], grid[r][c + 3]], 2);
    }
  }
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c < C4_COLS; c++) {
      score += c4ScoreWindow([grid[r][c], grid[r + 1][c], grid[r + 2][c], grid[r + 3][c]], 2);
    }
  }
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      score += c4ScoreWindow([grid[r][c], grid[r + 1][c + 1], grid[r + 2][c + 2], grid[r + 3][c + 3]], 2);
    }
  }
  for (let r = 3; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      score += c4ScoreWindow([grid[r][c], grid[r - 1][c + 1], grid[r - 2][c + 2], grid[r - 3][c + 3]], 2);
    }
  }
  return score;
}

function c4Minimax(grid, depth, alpha, beta, maximizing) {
  if (c4CheckWin(grid, 2)) return 10000 + depth * 100;
  if (c4CheckWin(grid, 1)) return -10000 - depth * 100;

  const validCols = [3, 2, 4, 1, 5, 0, 6].filter((c) => grid[0][c] === 0);
  if (validCols.length === 0) return 0;
  if (depth === 0) return c4EvaluateBoard(grid);

  if (maximizing) {
    let best = -Infinity;
    for (const c of validCols) {
      const r = c4GetLowestRow(grid, c);
      grid[r][c] = 2;
      const val = c4Minimax(grid, depth - 1, alpha, beta, false);
      grid[r][c] = 0;
      if (val > best) best = val;
      if (best > alpha) alpha = best;
      if (alpha >= beta) break;
    }
    return best;
  } else {
    let best = Infinity;
    for (const c of validCols) {
      const r = c4GetLowestRow(grid, c);
      grid[r][c] = 1;
      const val = c4Minimax(grid, depth - 1, alpha, beta, true);
      grid[r][c] = 0;
      if (val < best) best = val;
      if (best < beta) beta = best;
      if (alpha >= beta) break;
    }
    return best;
  }
}

function spawnConnectFourWidget(container) {
  const wrap = document.createElement("div");
  wrap.id = "c4-widget";
  wrap.className = "c4-wrap";

  const bar = document.createElement("div");
  bar.className = "c4-bar";

  const meta = document.createElement("span");
  meta.className = "c4-meta";
  const statusEl = document.createElement("span");
  meta.innerHTML = `<span class="c4-dot you"></span>you · <span class="c4-dot ai"></span>bot · `;
  meta.appendChild(statusEl);

  const resetBtn = document.createElement("span");
  resetBtn.className = "agent-cmd";
  resetBtn.textContent = "reset";

  bar.appendChild(meta);
  bar.appendChild(resetBtn);

  const boardEl = document.createElement("div");
  boardEl.className = "c4-board";

  const heldDisc = document.createElement("div");
  heldDisc.className = "c4-held-disc hidden";
  boardEl.appendChild(heldDisc);

  // Build persistent 7x6 DOM grid so hover previews never interrupt active drop animations
  const cellEls = Array.from({ length: C4_ROWS }, () => new Array(C4_COLS));
  const discEls = Array.from({ length: C4_ROWS }, () => new Array(C4_COLS));

  let state;

  function updateCursorDisc(e, colIdx) {
    if (!state || state.gameOver || state.turn !== 1 || colIdx < 0) {
      heldDisc.classList.add("hidden");
      return;
    }
    const dropRow = c4GetLowestRow(state.grid, colIdx);
    if (dropRow === -1) {
      heldDisc.classList.add("hidden");
      return;
    }
    if (e && typeof e.clientY === "number" && cellEls[dropRow][colIdx].getBoundingClientRect) {
      const boardRect = boardEl.getBoundingClientRect();
      const targetRect = cellEls[dropRow][colIdx].getBoundingClientRect();
      const topRect = cellEls[0][colIdx].getBoundingClientRect();
      const rPx = targetRect.height / 2 || 12;

      const minY = topRect.top + rPx;
      const maxY = targetRect.top + rPx;
      const clampedY = Math.max(minY, Math.min(maxY, e.clientY));

      heldDisc.style.left = `${targetRect.left - boardRect.left}px`;
      heldDisc.style.top = `${clampedY - boardRect.top - rPx}px`;
      heldDisc.classList.remove("hidden");

      state.cursorDropOffset = Math.min(-4, Math.round(clampedY - maxY));
    } else {
      // Fallback for synthetic/headless events without bounding rects
      state.cursorDropOffset = -Math.max(12, dropRow * 30);
    }
  }

  boardEl.onmouseleave = () => {
    if (state) {
      state.hoverCol = -1;
      heldDisc.classList.add("hidden");
      updatePreview();
    }
  };

  for (let c = 0; c < C4_COLS; c++) {
    const colEl = document.createElement("div");
    colEl.className = "c4-col";
    colEl.onmouseenter = (e) => {
      if (state && !state.gameOver && state.turn === 1) {
        state.hoverCol = c;
        updatePreview();
        updateCursorDisc(e, c);
      }
    };
    colEl.onmousemove = (e) => {
      if (state && !state.gameOver && state.turn === 1) {
        if (state.hoverCol !== c) {
          state.hoverCol = c;
          updatePreview();
        }
        updateCursorDisc(e, c);
      }
    };
    colEl.onmouseleave = () => {
      if (state && state.hoverCol === c) {
        state.hoverCol = -1;
        heldDisc.classList.add("hidden");
        updatePreview();
      }
    };
    colEl.onclick = (e) => onColClick(c, e);

    for (let r = 0; r < C4_ROWS; r++) {
      const cell = document.createElement("div");
      cell.className = "c4-cell";
      const disc = document.createElement("div");
      disc.className = "c4-disc";
      cell.appendChild(disc);
      colEl.appendChild(cell);
      cellEls[r][c] = cell;
      discEls[r][c] = disc;
    }
    boardEl.appendChild(colEl);
  }

  wrap.appendChild(bar);
  wrap.appendChild(boardEl);
  container.appendChild(wrap);

  function resetState() {
    state = {
      grid: Array.from({ length: C4_ROWS }, () => new Array(C4_COLS).fill(0)),
      turn: 1,
      hoverCol: -1,
      cursorDropOffset: -40,
      lastMove: null,
      animMove: null,
      winCells: null,
      gameOver: false,
    };
    heldDisc.classList.add("hidden");
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        cellEls[r][c].className = "c4-cell";
        discEls[r][c].className = "c4-disc";
        discEls[r][c].removeAttribute("style");
      }
    }
    renderBoard();
  }

  resetBtn.onclick = resetState;

  function logC4ToolCall(msg) {
    const log = document.getElementById("tool-calls-log");
    if (!log) return;
    log.innerHTML = "";
    const line = document.createElement("div");
    line.textContent = msg;
    log.appendChild(line);
  }

  function isWinCell(r, c) {
    if (!state.winCells) return false;
    return state.winCells.some(([wr, wc]) => wr === r && wc === c);
  }

  function updatePreview() {
    const previewRow =
      !state.gameOver && state.turn === 1 && state.hoverCol >= 0
        ? c4GetLowestRow(state.grid, state.hoverCol)
        : -1;
    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const isPrev = state.grid[r][c] === 0 && c === state.hoverCol && r === previewRow;
        cellEls[r][c].classList.toggle("preview", isPrev);
      }
    }
  }

  function aiStep() {
    if (state.gameOver) return;
    const t0 = performance.now();
    const validCols = [3, 2, 4, 1, 5, 0, 6].filter((c) => state.grid[0][c] === 0);
    if (validCols.length === 0) return;

    let bestCol = validCols[0];
    let bestScore = -Infinity;

    for (const c of validCols) {
      const r = c4GetLowestRow(state.grid, c);
      state.grid[r][c] = 2;
      if (c4CheckWin(state.grid, 2)) {
        bestCol = c;
        bestScore = 9999;
        state.grid[r][c] = 0;
        break;
      }
      const score = c4Minimax(state.grid, 3, -Infinity, Infinity, false) + (Math.random() * 4 - 2);
      state.grid[r][c] = 0;

      if (score > bestScore) {
        bestScore = score;
        bestCol = c;
      }
    }

    const dropRow = c4GetLowestRow(state.grid, bestCol);
    state.grid[dropRow][bestCol] = 2;
    state.lastMove = [dropRow, bestCol];
    state.animMove = { r: dropRow, c: bestCol, dropY: -Math.max(8, dropRow * 30) };

    const dt = (performance.now() - t0).toFixed(2);
    const evalStr = bestScore >= 0 ? `+${Math.round(bestScore)}` : `${Math.round(bestScore)}`;
    logC4ToolCall(`↳ [int8 · ${dt}ms] connect4_agent_move({ col: ${bestCol + 1}, eval: ${evalStr} })`);

    const winLine = c4CheckWin(state.grid, 2);
    if (winLine) {
      state.winCells = winLine;
      state.gameOver = true;
      statusEl.textContent = "bot wins";
    } else if (state.grid[0].every((v) => v !== 0)) {
      state.gameOver = true;
      statusEl.textContent = "draw";
    } else {
      state.turn = 1;
    }
    renderBoard();
  }

  function onColClick(col, e) {
    if (state.gameOver || state.turn !== 1) return;
    const dropRow = c4GetLowestRow(state.grid, col);
    if (dropRow === -1) return;

    updateCursorDisc(e, col);
    const dropY = state.cursorDropOffset || -Math.max(12, dropRow * 30);
    heldDisc.classList.add("hidden");

    state.grid[dropRow][col] = 1;
    state.lastMove = [dropRow, col];
    state.animMove = { r: dropRow, c: col, dropY };

    const winLine = c4CheckWin(state.grid, 1);
    if (winLine) {
      state.winCells = winLine;
      state.gameOver = true;
      statusEl.textContent = "you win!";
      renderBoard();
      return;
    }

    if (state.grid[0].every((v) => v !== 0)) {
      state.gameOver = true;
      statusEl.textContent = "draw";
      renderBoard();
      return;
    }

    state.turn = 2;
    renderBoard();
    const dropMs = Math.max(90, Math.min(250, Math.round(Math.abs(dropY) * 1.3 + 75)));
    setTimeout(aiStep, dropMs + 40);
  }

  function renderBoard() {
    if (!state.gameOver) {
      statusEl.textContent = state.turn === 1 ? "your turn" : "thinking...";
    }

    updatePreview();

    for (let r = 0; r < C4_ROWS; r++) {
      for (let c = 0; c < C4_COLS; c++) {
        const cell = cellEls[r][c];
        const disc = discEls[r][c];
        const val = state.grid[r][c];

        if (val === 1) {
          disc.classList.add("you");
        } else if (val === 2) {
          disc.classList.add("ai");
        }

        if (state.animMove && state.animMove.r === r && state.animMove.c === c) {
          const dy = state.animMove.dropY;
          const dms = Math.max(90, Math.min(250, Math.round(Math.abs(dy) * 1.3 + 75)));
          disc.style.setProperty("--drop-y", `${dy}px`);
          disc.style.setProperty("--drop-ms", `${dms}ms`);
          disc.classList.remove("dropping");
          void disc.offsetWidth;
          disc.classList.add("dropping");
        }

        const isLast =
          state.lastMove && state.lastMove[0] === r && state.lastMove[1] === c && !state.winCells;
        disc.classList.toggle("last-move", Boolean(isLast));

        const isDim = Boolean(state.winCells && !isWinCell(r, c));
        cell.classList.toggle("dim", isDim);
      }
    }
    state.animMove = null;
  }

  resetState();
}

/* =========================================================
 * MAIN AGENT EXECUTION & AUTOREGRESSIVE DECODING LOOP
 * ========================================================= */
async function runAgentQuery(e) {
  if (e) e.preventDefault();
  const inputEl = document.getElementById("agent-input");
  const q = inputEl.value.trim();
  if (!q) return;

  lastQueryCmd = q.toLowerCase();
  inputEl.value = "";
  const runId = ++currentRunId;

  const container = document.getElementById("agent-trace-container");
  const log = document.getElementById("tool-calls-log");
  const answerEl = document.getElementById("agent-answer");
  const clearBtn = document.getElementById("agent-clear");

  clearExtras();

  container.classList.remove("hidden");
  clearBtn.classList.remove("hidden");
  log.innerHTML = "";
  answerEl.textContent = "";
  clearHighlights();

  const { intent, prob, dtMs } = inferPromptIntent(q);
  const { words, genMs } = generateWithMicroGPT(intent);

  let toolCall = null;
  let showHelp = false;
  let showGame = false;

  switch (intent) {
    case "help_menu":
      showHelp = true;
      break;
    case "play_game":
      toolCall = `launch_game({ game: "connect_four" })`;
      showGame = true;
      break;
    case "chef_recipe":
      toolCall = `chef_recommend()`;
      break;
    case "explain_experience":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(EXP_IDS, "extended");
      focusRows(EXP_IDS);
      toolCall = `expand_section({ section: "experience" })`;
      break;
    case "explain_projects":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(PROJ_IDS, "extended");
      focusRows(PROJ_IDS);
      toolCall = `expand_section({ section: "projects" })`;
      break;
    case "explain_all":
      mutateRowsDetail(ALL_IDS, "extended");
      toolCall = `mutate_dom_copy({ depth: "extended" })`;
      break;
    case "set_detail_simple":
      collapseAllRows();
      toolCall = `mutate_dom_copy({ depth: "simple" })`;
      break;
    case "set_theme_dark":
      mutateTheme("dark");
      toolCall = `mutate_theme({ preset: "dark" })`;
      break;
    case "set_theme_light":
      mutateTheme("light");
      toolCall = `mutate_theme({ preset: "light" })`;
      break;
    case "set_theme_paper":
      mutateTheme("paper");
      toolCall = `mutate_theme({ preset: "paper" })`;
      break;
    case "set_theme_terminal":
      mutateTheme("terminal");
      toolCall = `mutate_theme({ preset: "terminal" })`;
      break;
    case "meta_self":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(["proj-tinygpt"], "extended");
      focusRows(["proj-tinygpt"]);
      toolCall = `focus_node({ id: "#proj-tinygpt" })`;
      break;
    case "focus_rag_agents":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(["exp-uoft", "proj-chorus", "proj-arch", "proj-tinygpt"], "extended");
      focusRows(["exp-uoft", "proj-chorus", "proj-arch", "proj-tinygpt"]);
      toolCall = `filter_dom_rows({ tag: "agents_and_rag" })`;
      break;
    case "focus_swe":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(["exp-mpbsdp", "exp-mgcs"], "extended");
      focusRows(["exp-mpbsdp", "exp-mgcs"]);
      toolCall = `filter_dom_rows({ tag: "fullstack_swe" })`;
      break;
    case "inspect_chorus":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(["proj-chorus"], "extended");
      focusRows(["proj-chorus"]);
      toolCall = `focus_node({ id: "#proj-chorus" })`;
      break;
    case "inspect_arch":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(["proj-arch"], "extended");
      focusRows(["proj-arch"]);
      toolCall = `focus_node({ id: "#proj-arch" })`;
      break;
    case "inspect_ml_projects":
      mutateRowsDetail(ALL_IDS, "simple");
      mutateRowsDetail(["proj-tinygpt", "proj-transformer", "proj-speech"], "extended");
      focusRows(["proj-tinygpt", "proj-transformer", "proj-speech"]);
      toolCall = `focus_node({ ids: ["#proj-tinygpt", "#proj-transformer", "#proj-speech"] })`;
      break;
    case "copy_email":
      if (navigator.clipboard) navigator.clipboard.writeText("jeffersonn.liu@gmail.com");
      toolCall = `clipboard_write("jeffersonn.liu@gmail.com")`;
      break;
    case "open_resume":
      window.open("./assets/resume.pdf", "_blank");
      toolCall = `window_open("./assets/resume.pdf")`;
      break;
    case "reset_page":
      collapseAllRows();
      document.documentElement.removeAttribute("style");
      toolCall = `reset_dom_state()`;
      break;
    case "unmatched":
      showHelp = true;
      break;
  }

  const line = document.createElement("div");
  if (toolCall) {
    line.textContent = `↳ [micro-gpt int8 · ${words.length} tok in ${genMs}ms] ${toolCall}`;
  } else {
    line.textContent = `↳ [micro-gpt int8 · 48d 4h · ${words.length} tok in ${genMs}ms]`;
  }
  log.appendChild(line);

  for (let i = 0; i < words.length; i++) {
    await new Promise((r) => setTimeout(r, 16));
    if (runId !== currentRunId) return;
    answerEl.textContent += (i === 0 ? "" : " ") + words[i];
  }

  if (runId !== currentRunId) return;

  if (showHelp) {
    renderHelpList(container);
  }
  if (showGame) {
    spawnConnectFourWidget(container);
  }
}

/* =========================================================
 * CLICK-TO-EXPAND & CLICK-OFF-TO-COLLAPSE ROW INTERACTIONS
 * ========================================================= */
if (typeof document !== "undefined") {
  document.addEventListener("click", (e) => {
    if (!document.body.contains(e.target) || e.target.closest(".agent-cmd")) return;

    const clickedRow = e.target.closest(".row");
    if (clickedRow) {
      const rowId = clickedRow.id;
      if (!rowId) return;

      const isAlreadySoleExpanded = expandedRows.size === 1 && expandedRows.has(rowId);
      if (isAlreadySoleExpanded) {
        collapseAllRows();
        if (rowId === "proj-tinygpt" && lastQueryCmd === "tinygpt") {
          clearAgent();
        }
      } else {
        mutateRowsDetail(ALL_IDS, "simple");
        mutateRowsDetail([rowId], "extended");
        focusRows([rowId]);
        if (rowId === "proj-tinygpt") {
          triggerCmd("tinygpt");
        }
      }
      return;
    }

    // Clicked outside any .row: collapse expanded rows unless interacting with the agent box
    if (
      !e.target.closest(".agent-inline") &&
      !e.target.closest("#agent-trace-container") &&
      !e.target.closest("form")
    ) {
      if (expandedRows.size > 0) {
        collapseAllRows();
        if (lastQueryCmd === "tinygpt") {
          clearAgent();
        }
      }
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      clearAgent();
    }
  });
}
