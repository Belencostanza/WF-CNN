def load_config(file):
    with open(file, 'r') as f:
        line = f.readlines()

    # Crear un diccionario a partir de las líneas
    parameters = {}
    for lin in line:
        if "=" in lin:
            # Utiliza '=' como delimitador y elimina posibles espacios
            #try:
            key, value = map(str.strip, lin.strip().split("=", 1))
            parameters[key] = value
            #except ValueError:
            #    print("Error al dividir la línea:", repr(linea))
        #else:
        #    print("Línea sin el formato clave=valor:", repr(linea.strip()))  # strip() aquí

    return parameters

# Ejemplo de uso
#archivo_config = 'input.dict'
#parametros_leidos = cargar_config(archivo_config)

# Usa los parámetros leídos en tu código
#print("Parametros leidos:", parametros_leidos)
#print(parametros_leidos['name_train'])
