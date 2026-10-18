# --- GÖREV 1: UNICODE VE UTF-8 ---

cumle = "çalıştığım öğretmen şüpheli"

print(f"{'Harf':<6} | {'Unicode (Codepoint)':<20} | {'UTF-8 (Hex)':<15} | {'Byte Boyutu'}")
print("-" * 60)

for harf in cumle:
    kod_noktasi = f"U+{ord(harf):04X}"
    utf8_byte = harf.encode("utf-8")
    hex_hali = utf8_byte.hex(" ")
    boyut = len(utf8_byte)
    print(f"{harf:<6} | {kod_noktasi:<20} | {hex_hali:<15} | {boyut} byte")