from PIL import Image
img = Image.open('financial_llm_governance_architecture.png')
w, h = img.size
legend = img.crop((0, int(h*0.78), w, h))
legend.save('legend_preview.png')
print(w, h)
