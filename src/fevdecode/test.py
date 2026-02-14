from parser import parse_fev_riff
info = parse_fev_riff(r"D:\Dev\Python\fevdecode\sound\sora.fev")
print(info.chunks)