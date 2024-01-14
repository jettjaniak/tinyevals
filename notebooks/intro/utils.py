import torch; torch.set_grad_enabled(False)
from datasets import load_dataset
from tqdm.auto import tqdm

# from notebook 00

ALLOWED_CHARS = set(" \n\"\'(),.:?!0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

def load_orig_ds_txt(split: str) -> list[str]:
    # checking just startswith, because you can include slice like "train[:1000]"
    assert split.startswith("train") or split.startswith("validation")
    hf_ds = load_dataset(f"roneneldan/TinyStories", split=split)
    dataset = []
    for sample_txt in tqdm(hf_ds["text"]):
        # encoding issues and rare weird prompts
        if not set(sample_txt).issubset(ALLOWED_CHARS):
            continue
        dataset.append(sample_txt)
    return dataset

# from notebook 01

def tokenize(tokenizer, sample_txt: str) -> list[int]:
    # supposedly this can be different than prepending the bos token id
    return tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0]

# from notebook 02

def get_logits(model, sample_tok):
    sample_tok = sample_tok.unsqueeze(0)
    return model(sample_tok).logits[0]

def get_correct_probs(model, sample_tok):
    # logits: pos, d_vocab
    logits = get_logits(model, sample_tok)
    # pos, d_vocab
    probs = torch.softmax(logits, dim=-1)
    # drop the value for the last position, as we don't know
    # what is the correct next token there
    probs = probs[:-1]
    # out of d_vocab values, take the one that corresponds to the correct next token
    return probs[range(len(probs)), sample_tok[1:]]

# from notebook 03

TOKEN_STYLE = {
    "border": "1px solid #888",
    "display": "inline-block",
    # each character of the same width, so we can easily spot a space
    "font-family": "monospace",
    "font-size": "14px",
    "color": "black",
    "background-color": "white",
    "margin": "1px 0px 1px 1px",
    "padding": "0px 1px 1px 1px",
}
TOKEN_STYLE_STR = " ".join([f"{k}: {v};" for k, v in TOKEN_STYLE.items()])
# every element of class "token" will have this style applied
STYLE_TAG = f"<style>.token {{ {TOKEN_STYLE_STR} }} #hover_info {{ font-family: monospace }}</style>"
HOVER_JS_TAG = """
<script>
var token_divs = document.querySelectorAll('.token');
var hover_info = document.getElementById('hover_info');

token_divs.forEach(function(token_div) {
    token_div.addEventListener('mouseover', function(e) {
    
        hover_info.innerHTML = ""
        for( var d in this.dataset) {
            hover_info.innerHTML += "<b>" + d + "</b> ";
            hover_info.innerHTML += this.dataset[d] + "<br>";
        }

        var curr_height = hover_info.clientHeight;
        var style_height_str = hover_info.style.minHeight;
        var style_height = 0;
        if (style_height_str != "") {
            style_height = parseInt(style_height_str.slice(0, -2));
        }
        if (curr_height > style_height) {
            hover_info.style.minHeight = curr_height + "px";
        }

    });

    token_div.addEventListener('mouseout', function(e) {
        hover_info.innerHTML = ""
    });
});
</script>
"""

def token_to_html(tokenizer, token, bg_color=None, hover_data=None):
    hover_data = hover_data or {}  # equivalent to `if not data: data = {}`
    # 1. non-breakable space, w/o it leading spaces wouldn't be displayed
    # 2. replace new line character with two characters: \ and n
    str_token = tokenizer.decode(token).replace(" ", "&nbsp;").replace("\n", r"\n")

    # background or user-select (for \n) goes here
    specific_styles = {}
    # optional line break
    br = ""

    if bg_color:
        specific_styles["background-color"] = bg_color
    if str_token == r"\n":
        # add line break in html
        br += "<br>"
        # this is so we can copy the prompt without "\n"s
        specific_styles["user-select"] = "none"

    style_str = data_str = ""
    # converting style dict into the style attribute
    if specific_styles:
        inside_style_str = "; ".join(f"{k}: {v}" for k, v in specific_styles.items())
        style_str = f" style='{inside_style_str}'"
    # converting data dict into data attributes
    if hover_data:
        data_str = "".join(f" data-{k}='{v.replace(' ', '&nbsp;')}'" for k, v in hover_data.items())
    return f"<div class='token'{style_str}{data_str}>{str_token}</div>{br}"

def vis_tokens(tokenizer, tokens, colors=None, hover_datas=None):
    token_htmls = []
    for i in range(len(tokens)):
        token = tokens[i]
        color = colors[i] if colors else None
        hover_data = hover_datas[i] if hover_datas else None
        token_htmls.append(token_to_html(tokenizer, token, color, hover_data))
    return STYLE_TAG + HOVER_JS_TAG + "".join(token_htmls) + "<div id='hover_info'></div>"

def probs_to_colors(probs):
    # for the endoftext token
    # no prediction, no color
    colors = ["white"]
    for p in probs.tolist():
        red_gap = 150  # the higher it is, the less red the tokens will be
        green_blue_val = red_gap + int((255 - red_gap) * (1 - p))
        colors.append(f"rgb(255, {green_blue_val}, {green_blue_val})")
    return colors