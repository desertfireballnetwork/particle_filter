#!/bin/bash
# concatonate fits HDU tables to one

# argument parsing
while getopts f:h opt; do
  case $opt in
    f)
      targetFile=$OPTARG
      ;;
    h)
      usage
      ;;
  esac
done


if [ ! -f "$targetFile" ]; then
  echo "file does NOT exist... ABORTING"
  exit 1
fi

echo "$(dirname "$targetFile")/$(basename "$targetFile" .fits)_all.fits"

# run tcat
stilts tcat in="$targetFile" multi=true out="$(dirname "$targetFile")/$(basename "$targetFile" .fits)_all.fits"

echo "$targetFile"
echo "has been written to"
echo "$(dirname "$targetFile")/$(basename "$targetFile" .fits)_all.fits"

