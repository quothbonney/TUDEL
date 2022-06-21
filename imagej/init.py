import imagej
from scyjava import jimport

# initialize ImageJ2 with Fiji plugins
ij = imagej.init('sc.fiji:fiji')

macro = """
#@ String name
#@ int age
#@ String city
#@output Object greeting
greeting = "Hello " + name + ". You are " + age + " years old, and live in " + city + "."
"""
args = {
    'name': 'Chuckles',
    'age': 13,
    'city': 'Nowhere'
}

language_extension = 'ijm'
result_script = ij.py.run_script(language_extension, macro, args)


ij.py.run_macro("""run("Blobs (25K)");""")
blobs = ij.WindowManager.getCurrentImage()
print(blobs)

ij.py.show(blobs)

# NB: This is not a built-in ImageJ command! It is the
# Plugins › Integral Image Filters › Mean command,
# which is part of mpicbg_, which is included with Fiji.
plugin = 'Mean'
args = {
    'block_radius_x': 10,
    'block_radius_y': 10
}
ij.py.run_plugin(plugin, args)


result = ij.WindowManager.getCurrentImage()
result = ij.py.show(result)

