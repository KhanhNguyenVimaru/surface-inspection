CLASS_MAP = {
    "Crazing": "Cr",
    "Inclusion": "In",
    "Patches": "Pa",
    "Pitted_surface": "Ps",
    "Rolled-in_scale": "Rs",
    "Scratches": "Sc",
}

IDX_TO_CODE = ["Cr", "In", "Pa", "Ps", "Rs", "Sc"]
CODE_TO_IDX = {code: idx for idx, code in enumerate(IDX_TO_CODE)}

IMG_SIZE = 224
