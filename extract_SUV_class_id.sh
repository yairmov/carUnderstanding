cat cars_class_id_class_name.txt | tail -196 | grep -i SUV | awk '{print $2}'
